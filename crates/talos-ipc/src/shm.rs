//! 共享内存区域封装 (基于 memmap2)
//!
//! 使用内存映射文件替代 POSIX 共享内存，纯 Rust 实现。

use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::io;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static REGION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum ShmError {
    IoError(io::Error),
    MapFailed,
    InvalidSize,
}

impl std::fmt::Display for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmError::IoError(e) => write!(f, "IO error: {}", e),
            ShmError::MapFailed => write!(f, "mmap failed"),
            ShmError::InvalidSize => write!(f, "invalid size"),
        }
    }
}

impl std::error::Error for ShmError {}

impl From<io::Error> for ShmError {
    fn from(e: io::Error) -> Self {
        ShmError::IoError(e)
    }
}

/// 获取共享内存文件路径
fn shm_path(name: &str) -> PathBuf {
    // 使用 /tmp 目录，移除开头的 '/'
    let clean_name = name.trim_start_matches('/');
    PathBuf::from("/tmp").join(clean_name)
}

/// RAII 封装的共享内存区域
pub struct ShmRegion {
    mmap: MmapMut,
    path: PathBuf,
    owner_lock: Option<File>,
    #[cfg(unix)]
    identity: (u64, u64),
}

unsafe impl Send for ShmRegion {}
unsafe impl Sync for ShmRegion {}

impl ShmRegion {
    /// 创建新的共享内存区域 (生产者)
    pub fn create(name: &str, size: usize) -> Result<Self, ShmError> {
        if size == 0 {
            return Err(ShmError::InvalidSize);
        }
        let path = shm_path(name);
        let lock_path = path.with_file_name(format!(
            ".{}.lock",
            path.file_name()
                .ok_or(ShmError::InvalidSize)?
                .to_string_lossy()
        ));
        // Keep this lock inode stable across publisher generations. In particular an old
        // owner's Drop must not race a new owner's rename between stat and unlink.
        let owner_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        owner_lock.lock()?;

        let (temporary_path, file) = loop {
            let sequence = REGION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary_path = path.with_file_name(format!(
                ".{}.{}.{}.new",
                path.file_name().unwrap().to_string_lossy(),
                std::process::id(),
                sequence
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&temporary_path)
            {
                Ok(file) => break (temporary_path, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        };

        let initialized = (|| -> Result<_, ShmError> {
            // A fresh file is zero-filled by set_len. Never truncate the inode that
            // existing consumers have mapped: doing so can SIGBUS those processes.
            file.set_len(size as u64)?;
            let mmap = unsafe { MmapMut::map_mut(&file)? };
            #[cfg(unix)]
            let identity = {
                let metadata = file.metadata()?;
                (metadata.dev(), metadata.ino())
            };
            std::fs::rename(&temporary_path, &path)?;
            owner_lock.unlock()?;
            Ok(Self {
                mmap,
                path,
                owner_lock: Some(owner_lock),
                #[cfg(unix)]
                identity,
            })
        })();
        if initialized.is_err() {
            let _ = std::fs::remove_file(temporary_path);
        }
        initialized
    }

    /// 打开已存在的共享内存区域 (消费者)
    pub fn open(name: &str, size: usize) -> Result<Self, ShmError> {
        if size == 0 {
            return Err(ShmError::InvalidSize);
        }
        let path = shm_path(name);

        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        // 验证大小
        let metadata = file.metadata()?;
        if metadata.len() < size as u64 {
            return Err(ShmError::InvalidSize);
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            path,
            owner_lock: None,
            #[cfg(unix)]
            identity: (metadata.dev(), metadata.ino()),
        })
    }

    /// 获取指向共享内存的指针
    pub fn as_ptr(&self) -> *mut u8 {
        self.mmap.as_ptr() as *mut u8
    }

    /// 获取共享内存大小
    pub fn size(&self) -> usize {
        self.mmap.len()
    }

    /// 将共享内存解释为指定类型
    ///
    /// # Safety
    /// 调用者必须确保类型 T 与共享内存布局匹配
    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { &*(self.mmap.as_ptr() as *const T) }
    }

    /// 将共享内存解释为指定类型 (可变)
    ///
    /// # Safety
    /// 调用者必须确保类型 T 与共享内存布局匹配
    pub unsafe fn as_mut<T>(&mut self) -> &mut T {
        unsafe { &mut *(self.mmap.as_ptr() as *mut T) }
    }

    /// 刷新内存映射到文件
    pub fn flush(&self) -> Result<(), ShmError> {
        self.mmap.flush()?;
        Ok(())
    }
}

impl Drop for ShmRegion {
    fn drop(&mut self) {
        if let Some(lock) = &self.owner_lock {
            if lock.lock().is_err() {
                return;
            }
            #[cfg(unix)]
            if std::fs::metadata(&self.path)
                .is_ok_and(|metadata| (metadata.dev(), metadata.ino()) == self.identity)
            {
                let _ = std::fs::remove_file(&self.path);
            }
            // Non-Unix platforms leave cleanup to the next atomic replacement;
            // without an inode identity comparison deletion is unsafe.
            let _ = lock.unlock();
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn unique_name() -> String {
        format!(
            "talos_restart_test_{}_{}",
            std::process::id(),
            REGION_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn remove_lock(name: &str) {
        let _ = std::fs::remove_file(format!("/tmp/.{name}.lock"));
    }

    #[test]
    fn restart_replaces_inode_without_invalidating_old_mapping() {
        let name = unique_name();
        let old = ShmRegion::create(&name, 8192).unwrap();
        unsafe { old.as_ptr().add(8191).write(73) };
        let reader = ShmRegion::open(&name, 8192).unwrap();
        let new = ShmRegion::create(&name, 4096).unwrap();
        assert_ne!(old.identity, new.identity);
        assert_eq!(reader.size(), 8192);
        assert_eq!(unsafe { reader.as_ptr().add(8191).read() }, 73);
        assert_eq!(unsafe { new.as_ptr().read() }, 0);
        unsafe { new.as_ptr().write(42) };
        let new_reader = ShmRegion::open(&name, 4096).unwrap();
        assert_eq!(unsafe { new_reader.as_ptr().read() }, 42);

        // The old publisher can exit after its replacement has already started.
        drop(old);
        assert!(shm_path(&name).exists());
        assert_eq!(unsafe { reader.as_ptr().add(8191).read() }, 73);
        drop(new);
        assert!(!shm_path(&name).exists());
        assert_eq!(unsafe { new_reader.as_ptr().read() }, 42);
        drop((reader, new_reader));
        remove_lock(&name);
    }

    #[test]
    fn consumer_drop_preserves_owner_and_sizes_are_checked() {
        let name = unique_name();
        assert!(matches!(
            ShmRegion::create(&name, 0),
            Err(ShmError::InvalidSize)
        ));
        let owner = ShmRegion::create(&name, 4096).unwrap();
        assert!(matches!(
            ShmRegion::open(&name, 8192),
            Err(ShmError::InvalidSize)
        ));
        assert!(matches!(
            ShmRegion::open(&name, 0),
            Err(ShmError::InvalidSize)
        ));
        drop(ShmRegion::open(&name, 4096).unwrap());
        assert!(shm_path(&name).exists());
        drop(owner);
        assert!(!shm_path(&name).exists());
        remove_lock(&name);
    }
}
