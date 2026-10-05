//! The fold's secret wrapper — zeroize-on-drop, redacted `Debug`, no foreign crate.
//!
//! The tasmanian-cloud crates wrapped credentials in `secrecy::SecretString` /
//! `secrecy::SecretBox` for two properties: the buffer is wiped on drop and `Debug`
//! never prints the value. Both belong in the substrate, not in a dependency: this
//! module is the fold form, with the same surface (`expose_secret`) so call sites
//! change only their import.

use std::fmt;

/// A string whose heap buffer is zeroed on drop and whose `Debug` is redacted.
///
/// `Clone` exists for parity with the wrapper it replaces, but prefer moving:
/// every clone is a second copy of the secret in memory.
#[derive(Clone)]
pub struct SecretString {
    inner: String,
}

impl SecretString {
    /// The secret value. Name matches the wrapper being replaced so call sites
    /// change only their import.
    pub fn expose_secret(&self) -> &str {
        &self.inner
    }

    /// Consume and return the value — use only at the boundary that needs it.
    /// The wrapper is `Drop`, so the inner string is taken rather than moved;
    /// the emptied shell still runs its zeroing drop.
    pub fn into_inner(mut self) -> String {
        std::mem::take(&mut self.inner)
    }
}

impl From<String> for SecretString {
    fn from(inner: String) -> Self {
        Self { inner }
    }
}

impl From<&str> for SecretString {
    fn from(inner: &str) -> Self {
        Self { inner: inner.to_string() }
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(<redacted>)")
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        for byte in unsafe { self.inner.as_bytes_mut() } {
            *byte = 0;
        }
    }
}

/// A byte slice whose buffer is zeroed on drop and whose `Debug` is redacted.
#[derive(Clone)]
pub struct SecretBox {
    inner: Box<[u8]>,
}

impl SecretBox {
    /// Wrap bytes that must not survive the process.
    pub fn new(data: Box<[u8]>) -> Self {
        Self { inner: data }
    }

    /// The secret value. Name matches the wrapper being replaced.
    pub fn expose_secret(&self) -> &[u8] {
        &self.inner
    }
}

impl fmt::Debug for SecretBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretBox(<redacted>)")
    }
}

impl Drop for SecretBox {
    fn drop(&mut self) {
        for byte in self.inner.iter_mut() {
            *byte = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_never_prints_the_value() {
        let secret = SecretString::from("hunter2");
        let rendered = format!("{secret:?}");
        assert!(!rendered.contains("hunter2"));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn secret_string_round_trips_the_value() {
        let secret = SecretString::from("token");
        assert_eq!(secret.expose_secret(), "token");
        assert_eq!(SecretString::from("t").into_inner(), "t");
    }

    #[test]
    fn secret_box_exposes_bytes_and_redacts_debug() {
        let secret = SecretBox::new(vec![1u8, 2, 3].into_boxed_slice());
        assert_eq!(secret.expose_secret(), &[1, 2, 3]);
        assert!(!format!("{secret:?}").contains('1'));
    }
}
