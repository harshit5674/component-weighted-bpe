use super::*;
use crate::StreamError;

/// The stored state of one prefix: its last token and the endpoint of the
/// prefix that precedes that token (`None` for the empty prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prefix {
    pub last: Option<TokenId>,
    pub parent: Option<usize>,
}

#[derive(Clone, Copy)]
pub(super) struct Record {
    last: Option<TokenId>,
    parent: usize,
    pub(super) dfs: usize,
}

/// Maintains the exact BPE last token of every prefix of a growing byte
/// string. Reserve capacity before appending; successful appends allocate
/// nothing. The count-balanced and component-weighted searches take
/// O(log² t) and O(log t) worst-case time per append.
pub struct CentroidStream {
    compiled: Arc<CentroidCompiled>,
    ac_state: usize,
    records: Vec<Record>,
    capacity: usize,
}

impl CentroidStream {
    pub fn with_capacity(
        compiled: Arc<CentroidCompiled>,
        max_bytes: usize,
    ) -> Result<Self, StreamError> {
        let count = max_bytes.checked_add(1).ok_or(StreamError::Overflow)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(count)
            .map_err(|_| StreamError::Allocation)?;
        records.push(Record {
            last: None,
            parent: 0,
            dfs: 0,
        });
        Ok(Self {
            compiled,
            ac_state: 0,
            records,
            capacity: max_bytes,
        })
    }
    pub fn len(&self) -> usize {
        self.records.len() - 1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn remaining_capacity(&self) -> usize {
        self.capacity - self.len()
    }
    pub fn compiled(&self) -> &CentroidCompiled {
        &self.compiled
    }

    /// Explicit history growth, outside the reserved append operation.
    pub fn reserve(&mut self, additional: usize) -> Result<(), StreamError> {
        let required = self
            .len()
            .checked_add(additional)
            .ok_or(StreamError::Overflow)?;
        if required <= self.capacity {
            return Ok(());
        }
        let count = required.checked_add(1).ok_or(StreamError::Overflow)?;
        self.records
            .try_reserve_exact(count - self.records.len())
            .map_err(|_| StreamError::Allocation)?;
        self.capacity = required;
        Ok(())
    }
    pub fn append(&mut self, byte: u8) -> Result<TokenId, StreamError> {
        self.append_inner::<false>(byte).map(|(t, _)| t)
    }
    /// Runs the same search with per-operation counters. Ordinary append uses a
    /// separate const-generic specialization with counter updates compiled out.
    pub fn append_profiled(&mut self, byte: u8) -> Result<(TokenId, SearchStats), StreamError> {
        self.append_inner::<true>(byte)
    }
    fn append_inner<const TRACK: bool>(
        &mut self,
        byte: u8,
    ) -> Result<(TokenId, SearchStats), StreamError> {
        let atom = self
            .compiled
            .dictionary
            .atom(byte)
            .ok_or(StreamError::UnknownByte(byte))?;
        if self.len() == self.capacity {
            return Err(StreamError::Capacity {
                needed: 1,
                available: 0,
            });
        }
        let state = self.compiled.go[self.ac_state * self.compiled.alphabet_size + atom as usize];
        let matched = self.compiled.longest[state];
        let mut stats = SearchStats::default();
        let h = match self.compiled.report.search {
            CentroidSearch::CountBalanced => {
                self.compiled
                    .last::<false, TRACK>(matched, &self.records, &mut stats)
            }
            CentroidSearch::ComponentWeighted => {
                self.compiled
                    .last::<true, TRACK>(matched, &self.records, &mut stats)
            }
        };
        let token = self.compiled.tokens[h as usize];
        self.records.push(Record {
            last: Some(h),
            parent: self.records.len() - token.len,
            dfs: token.dfs,
        });
        self.ac_state = state;
        Ok((h, stats))
    }
    /// The whole chunk is checked before any byte changes the stream.
    pub fn extend(&mut self, bytes: &[u8]) -> Result<(), StreamError> {
        if bytes.len() > self.remaining_capacity() {
            return Err(StreamError::Capacity {
                needed: bytes.len(),
                available: self.remaining_capacity(),
            });
        }
        for &b in bytes {
            if self.compiled.dictionary.atom(b).is_none() {
                return Err(StreamError::UnknownByte(b));
            }
        }
        for &b in bytes {
            self.append(b)?;
        }
        Ok(())
    }
    pub fn prefix(&self, endpoint: usize) -> Option<Prefix> {
        self.records.get(endpoint).map(|r| Prefix {
            last: r.last,
            parent: (endpoint != 0).then_some(r.parent),
        })
    }
    pub fn reconstruct(&self, mut endpoint: usize) -> Result<Vec<TokenId>, StreamError> {
        if endpoint > self.len() {
            return Err(StreamError::InvalidPrefix {
                requested: endpoint,
                length: self.len(),
            });
        }
        let mut output = Vec::new();
        while endpoint != 0 {
            let r = self.records[endpoint];
            output.push(r.last.expect("nonempty prefix"));
            endpoint = r.parent;
        }
        output.reverse();
        Ok(output)
    }
    pub fn reset(&mut self) {
        self.records.truncate(1);
        self.ac_state = 0;
    }
}
