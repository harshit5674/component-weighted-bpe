use std::fmt;

/// Errors from dictionary validation and compilation. Construction never
/// substitutes a slower online algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidAlphabet,
    InvalidOperand {
        rule: usize,
        token: u32,
    },
    DuplicateSpelling {
        rule: usize,
    },
    NonCanonical {
        rule: usize,
        actual: Vec<u32>,
    },
    Limit {
        resource: &'static str,
        requested: usize,
        limit: usize,
    },
    Allocation,
    Overflow,
    Invariant(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlphabet => write!(f, "alphabet must contain 1 to 256 distinct bytes"),
            Self::InvalidOperand { rule, token } => {
                write!(f, "rule {rule} uses unavailable token {token}")
            }
            Self::DuplicateSpelling { rule } => {
                write!(f, "rule {rule} duplicates a token spelling")
            }
            Self::NonCanonical { rule, actual } => write!(
                f,
                "rule {rule} is not canonical: isolated pre-pass tokens are {actual:?}"
            ),
            Self::Limit {
                resource,
                requested,
                limit,
            } => write!(f, "{resource} requires {requested}, limit is {limit}"),
            Self::Allocation => write!(f, "allocation failed"),
            Self::Overflow => write!(f, "size arithmetic overflow"),
            Self::Invariant(message) => write!(f, "compiler invariant failed: {message}"),
        }
    }
}
impl std::error::Error for Error {}

/// Errors from stream operations. A failed append leaves the stream unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    UnknownByte(u8),
    Capacity { needed: usize, available: usize },
    InvalidPrefix { requested: usize, length: usize },
    Allocation,
    Overflow,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownByte(a) => write!(f, "byte {a} is outside the dictionary alphabet"),
            Self::Capacity { needed, available } => write!(
                f,
                "need capacity for {needed} more bytes; {available} available (reserve explicitly)"
            ),
            Self::InvalidPrefix { requested, length } => {
                write!(f, "prefix {requested} exceeds stream length {length}")
            }
            Self::Allocation => write!(f, "history allocation failed"),
            Self::Overflow => write!(f, "history size overflow"),
        }
    }
}
impl std::error::Error for StreamError {}

pub(crate) fn checked_product(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

pub(crate) fn limit(resource: &'static str, requested: usize, maximum: usize) -> Result<(), Error> {
    if requested > maximum {
        Err(Error::Limit {
            resource,
            requested,
            limit: maximum,
        })
    } else {
        Ok(())
    }
}
