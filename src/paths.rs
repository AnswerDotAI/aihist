//! Checked filesystem paths shared by history media and gateway file access.

use std::path::{Path, PathBuf};

/// Resolve existing symlinks and prospective targets without leaving the served root.
pub fn realpath(root: &Path, rel: &str) -> Result<PathBuf, String> {
    use std::path::Component::{CurDir, ParentDir, RootDir};
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let mut path = root.clone();
    for part in Path::new(rel).components() {
        match part {
            RootDir | CurDir => (),
            ParentDir => { path.pop(); }
            part => {
                path.push(part);
                match path.canonicalize() {
                    Ok(real) => path = real, Err(e) if e.kind() == std::io::ErrorKind::NotFound && !path.is_symlink() => (), Err(e) => return Err(e.to_string())
                }
            }
        }
        if !path.starts_with(&root) { return Err("path escapes the served root".into()); }
    }
    Ok(path)
}
