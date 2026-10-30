use crate::error::{checked_product, limit};
use crate::Error;

/// Charges compiled table payloads against a byte budget before allocating
/// them. This is not a peak-memory bound: scratch maps, vector slack and
/// allocator metadata are not charged.
pub(crate) struct Budget {
    max_table_bytes: usize,
    pub charged: usize,
}

impl Budget {
    pub fn new(max_table_bytes: usize) -> Self {
        Self {
            max_table_bytes,
            charged: 0,
        }
    }
    pub fn charge<T>(&mut self, len: usize) -> Result<(), Error> {
        let bytes = checked_product(len, std::mem::size_of::<T>())?;
        let total = self.charged.checked_add(bytes).ok_or(Error::Overflow)?;
        limit("compiled table payload bytes", total, self.max_table_bytes)?;
        self.charged = total;
        Ok(())
    }
    pub fn filled<T: Clone>(&mut self, len: usize, value: T) -> Result<Vec<T>, Error> {
        self.charge::<T>(len)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .map_err(|_| Error::Allocation)?;
        result.resize(len, value);
        Ok(result)
    }
}
