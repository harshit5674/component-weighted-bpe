// Unsafe code is confined to this test's counting allocator. The library forbids it.
#![allow(unsafe_code)]
use component_weighted_bpe::witness::{Letters, Witness};
use component_weighted_bpe::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

struct Counter;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: Counter = Counter;

// A single test in this process, so other test threads cannot touch the counter.
#[test]
fn reserved_appends_allocate_nothing() {
    let witness = Witness::new(6, Letters::DistinctPairs).unwrap();
    let dictionaries = [
        Arc::new(Dictionary::new(b"abc", &[(0, 1), (0, 3), (0, 4), (0, 2)], 1000).unwrap()),
        Arc::new(Dictionary::new(b"abc", &[(0, 0), (3, 1), (3, 4)], 1000).unwrap()),
        witness.dictionary.clone(),
    ];
    let inputs: [&[u8]; 3] = [b"abacabcabaab", b"aaabaaaabaab", &witness.input];
    let mut streams = Vec::new();
    for (dict, input) in dictionaries.iter().zip(inputs) {
        for search in [
            CentroidSearch::CountBalanced,
            CentroidSearch::ComponentWeighted,
        ] {
            let options = CentroidOptions {
                search,
                ..Default::default()
            };
            let compiled = Arc::new(CentroidCompiled::compile(dict.clone(), &options).unwrap());
            let stream = CentroidStream::with_capacity(compiled, 4 * input.len()).unwrap();
            streams.push((stream, input));
        }
    }
    ENABLED.store(true, Ordering::SeqCst);
    for (stream, input) in &mut streams {
        for &b in input.iter() {
            stream.append(b).unwrap();
        }
        for &b in input.iter() {
            stream.append_profiled(b).unwrap();
        }
        stream.extend(input).unwrap();
        let _ = stream.prefix(input.len());
        assert!(stream.append(b'z').is_err());
        stream.reset();
        stream.extend(input).unwrap();
    }
    ENABLED.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
