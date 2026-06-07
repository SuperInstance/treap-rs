//! A randomized treap — simultaneously a BST by key and a max-heap by priority.
//! Random priorities ensure O(log n) expected height.
//!
//! Core operations (split and merge) run in O(log n) expected time.
//! Insert, delete, contains, rank, and select are also O(log n) expected.

use std::cell::Cell;
use std::cmp::Ordering::*;

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

thread_local! {
    static RNG_STATE: Cell<u64> = const { Cell::new(0x517cc1b727220a95) };
}

fn next_priority() -> u64 {
    RNG_STATE.with(|state| {
        let mut s = state.get();
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        state.set(s);
        s
    })
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

type Link<K> = Option<Box<Node<K>>>;

struct Node<K: Ord> {
    key: K,
    priority: u64,
    left: Link<K>,
    right: Link<K>,
    size: usize,
}

fn node_size<K: Ord>(link: &Link<K>) -> usize {
    link.as_ref().map_or(0, |n| n.size)
}

fn update_size<K: Ord>(n: &mut Node<K>) {
    n.size = 1 + node_size(&n.left) + node_size(&n.right);
}

impl<K: Ord> Node<K> {
    fn new(key: K) -> Box<Self> {
        Box::new(Node {
            key,
            priority: next_priority(),
            left: None,
            right: None,
            size: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Core free functions: split, split_lt, merge
// ---------------------------------------------------------------------------

/// Split into (keys ≤ key, keys > key).
fn split<K: Ord>(root: Link<K>, key: &K) -> (Link<K>, Link<K>) {
    match root {
        None => (None, None),
        Some(mut node) => {
            if &node.key <= key {
                let (rl, rr) = split(node.right.take(), key);
                node.right = rl;
                update_size(&mut node);
                (Some(node), rr)
            } else {
                let (ll, lr) = split(node.left.take(), key);
                node.left = lr;
                update_size(&mut node);
                (ll, Some(node))
            }
        }
    }
}

/// Split into (keys < key, keys >= key).
fn split_lt<K: Ord>(root: Link<K>, key: &K) -> (Link<K>, Link<K>) {
    match root {
        None => (None, None),
        Some(mut node) => {
            if &node.key < key {
                let (rl, rr) = split_lt(node.right.take(), key);
                node.right = rl;
                update_size(&mut node);
                (Some(node), rr)
            } else {
                let (ll, lr) = split_lt(node.left.take(), key);
                node.left = lr;
                update_size(&mut node);
                (ll, Some(node))
            }
        }
    }
}

/// Merge two treaps where all keys in `left` < all keys in `right`.
fn merge<K: Ord>(left: Link<K>, right: Link<K>) -> Link<K> {
    match (left, right) {
        (None, r) => r,
        (l, None) => l,
        (Some(mut l), Some(mut r)) => {
            if l.priority >= r.priority {
                l.right = merge(l.right.take(), Some(r));
                update_size(&mut l);
                Some(l)
            } else {
                r.left = merge(Some(l), r.left.take());
                update_size(&mut r);
                Some(r)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Search helper
// ---------------------------------------------------------------------------

fn search_node<K: Ord>(link: &Link<K>, key: &K) -> bool {
    match link {
        None => false,
        Some(n) => match key.cmp(&n.key) {
            Less => search_node(&n.left, key),
            Greater => search_node(&n.right, key),
            Equal => true,
        },
    }
}

// ---------------------------------------------------------------------------
// Insert helper (returns new root + whether the key was newly inserted)
// ---------------------------------------------------------------------------

fn insert_node<K: Ord>(root: Link<K>, key: K) -> (Link<K>, bool) {
    if search_node(&root, &key) {
        return (root, false);
    }
    let new_node = Node::new(key);
    // Borrow new_node.key before it is moved
    let (left, right) = {
        let k = &new_node.key;
        split(root, k)
    };
    // left: keys <= new_node.key, but key wasn't present so keys < new_node.key
    let new_root = merge(merge(left, Some(new_node)), right);
    (new_root, true)
}

// ---------------------------------------------------------------------------
// Delete helper
// ---------------------------------------------------------------------------

fn delete_node<K: Ord>(root: Link<K>, key: &K) -> (bool, Link<K>) {
    match root {
        None => (false, None),
        Some(mut node) => match key.cmp(&node.key) {
            Less => {
                let (found, new_left) = delete_node(node.left.take(), key);
                node.left = new_left;
                update_size(&mut node);
                (found, Some(node))
            }
            Greater => {
                let (found, new_right) = delete_node(node.right.take(), key);
                node.right = new_right;
                update_size(&mut node);
                (found, Some(node))
            }
            Equal => (true, merge(node.left.take(), node.right.take())),
        },
    }
}

// ---------------------------------------------------------------------------
// Union helper
// ---------------------------------------------------------------------------

fn union_nodes<K: Ord>(t1: Link<K>, t2: Link<K>) -> Link<K> {
    match (t1, t2) {
        (None, t) => t,
        (t, None) => t,
        (Some(mut root), t2) => {
            // Split t2 into (keys < root.key) and (keys >= root.key)
            let (t2_left, t2_geq) = split_lt(t2, &root.key);
            // Drop any node in t2_geq that equals root.key
            let t2_right = remove_exact(t2_geq, &root.key);
            root.left = union_nodes(root.left.take(), t2_left);
            root.right = union_nodes(root.right.take(), t2_right);
            update_size(&mut root);
            Some(root)
        }
    }
}

/// Remove the node with exactly `key` from the tree (if present).
/// Used in union to eliminate duplicates.
fn remove_exact<K: Ord>(root: Link<K>, key: &K) -> Link<K> {
    match root {
        None => None,
        Some(mut node) => match key.cmp(&node.key) {
            Equal => merge(node.left.take(), node.right.take()),
            Less => {
                node.left = remove_exact(node.left.take(), key);
                update_size(&mut node);
                Some(node)
            }
            Greater => {
                node.right = remove_exact(node.right.take(), key);
                update_size(&mut node);
                Some(node)
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Intersection helper
// ---------------------------------------------------------------------------

fn intersect_nodes<K: Ord>(t1: Link<K>, t2: &Link<K>) -> Link<K> {
    match t1 {
        None => None,
        Some(mut node) => {
            let in_t2 = search_node(t2, &node.key);
            let new_left = intersect_nodes(node.left.take(), t2);
            let new_right = intersect_nodes(node.right.take(), t2);
            if in_t2 {
                node.left = new_left;
                node.right = new_right;
                update_size(&mut node);
                Some(node)
            } else {
                merge(new_left, new_right)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inorder traversal helper
// ---------------------------------------------------------------------------

fn inorder_collect<'a, K: Ord>(link: &'a Link<K>, out: &mut Vec<&'a K>) {
    if let Some(n) = link {
        inorder_collect(&n.left, out);
        out.push(&n.key);
        inorder_collect(&n.right, out);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A randomized treap: a BST by key and a max-heap by priority.
pub struct Treap<K: Ord> {
    root: Link<K>,
    size: usize,
}

impl<K: Ord> Treap<K> {
    /// Create an empty treap.
    pub fn new() -> Self {
        Treap { root: None, size: 0 }
    }

    /// Insert `key`. Returns `true` if the key was newly inserted, `false` if it was already present.
    pub fn insert(&mut self, key: K) -> bool {
        let (new_root, inserted) = insert_node(self.root.take(), key);
        self.root = new_root;
        if inserted {
            self.size += 1;
        }
        inserted
    }

    /// Delete `key`. Returns `true` if the key was found and removed.
    pub fn delete(&mut self, key: &K) -> bool {
        let (found, new_root) = delete_node(self.root.take(), key);
        self.root = new_root;
        if found {
            self.size -= 1;
        }
        found
    }

    /// Returns `true` if the treap contains `key`.
    pub fn contains(&self, key: &K) -> bool {
        search_node(&self.root, key)
    }

    /// Returns all keys in ascending (inorder) order.
    pub fn inorder(&self) -> Vec<&K> {
        let mut out = Vec::with_capacity(self.size);
        inorder_collect(&self.root, &mut out);
        out
    }

    /// Number of keys in the treap.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the treap contains no keys.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Split into `(left, right)` where `left` contains all keys ≤ `key`
    /// and `right` contains all keys > `key`.
    pub fn split(mut self, key: &K) -> (Treap<K>, Treap<K>) {
        let (l, r) = split(self.root.take(), key);
        let ls = node_size(&l);
        let rs = node_size(&r);
        (
            Treap { root: l, size: ls },
            Treap { root: r, size: rs },
        )
    }

    /// Merge two treaps. **Requires** every key in `left` to be strictly less
    /// than every key in `right`.
    pub fn merge(mut left: Treap<K>, mut right: Treap<K>) -> Treap<K> {
        let size = left.size + right.size;
        let root = merge(left.root.take(), right.root.take());
        Treap { root, size }
    }

    /// Return a new treap containing all keys from both treaps (no duplicates).
    pub fn union(mut self, mut other: Treap<K>) -> Treap<K> {
        let root = union_nodes(self.root.take(), other.root.take());
        let size = node_size(&root);
        Treap { root, size }
    }

    /// Return a new treap containing only keys present in both `self` and `other`.
    pub fn intersection(mut self, other: &Treap<K>) -> Treap<K> {
        let root = intersect_nodes(self.root.take(), &other.root);
        let size = node_size(&root);
        Treap { root, size }
    }

    /// Count of keys strictly less than `key` (0-based rank).
    pub fn rank(&self, key: &K) -> usize {
        fn rank_node<K: Ord>(link: &Link<K>, key: &K) -> usize {
            match link {
                None => 0,
                Some(n) => match key.cmp(&n.key) {
                    Less => rank_node(&n.left, key),
                    Greater => 1 + node_size(&n.left) + rank_node(&n.right, key),
                    Equal => node_size(&n.left),
                },
            }
        }
        rank_node(&self.root, key)
    }

    /// Return the `k`-th smallest key (0-indexed), or `None` if `k >= len()`.
    pub fn select(&self, k: usize) -> Option<&K> {
        fn select_node<K: Ord>(link: &Link<K>, k: usize) -> Option<&K> {
            match link {
                None => None,
                Some(n) => {
                    let ls = node_size(&n.left);
                    if k < ls {
                        select_node(&n.left, k)
                    } else if k == ls {
                        Some(&n.key)
                    } else {
                        select_node(&n.right, k - ls - 1)
                    }
                }
            }
        }
        select_node(&self.root, k)
    }
}

impl<K: Ord> Default for Treap<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_treap(keys: &[i32]) -> Treap<i32> {
        let mut t = Treap::new();
        for &k in keys {
            t.insert(k);
        }
        t
    }

    #[test]
    fn test_new_empty() {
        let t: Treap<i32> = Treap::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_insert_contains() {
        let mut t = Treap::new();
        t.insert(42);
        assert!(t.contains(&42));
    }

    #[test]
    fn test_insert_new_true() {
        let mut t = Treap::new();
        assert!(t.insert(10));
    }

    #[test]
    fn test_insert_dup_false() {
        let mut t = Treap::new();
        t.insert(10);
        assert!(!t.insert(10));
    }

    #[test]
    fn test_contains_miss_empty() {
        let t: Treap<i32> = Treap::new();
        assert!(!t.contains(&5));
    }

    #[test]
    fn test_contains_miss_after_inserts() {
        let t = make_treap(&[1, 2, 3]);
        assert!(!t.contains(&99));
    }

    #[test]
    fn test_inorder_empty() {
        let t: Treap<i32> = Treap::new();
        assert_eq!(t.inorder(), Vec::<&i32>::new());
    }

    #[test]
    fn test_inorder_single() {
        let t = make_treap(&[7]);
        assert_eq!(t.inorder(), vec![&7]);
    }

    #[test]
    fn test_inorder_sorted() {
        let t = make_treap(&[5, 3, 7, 1, 4, 6, 8]);
        assert_eq!(t.inorder(), vec![&1, &3, &4, &5, &6, &7, &8]);
    }

    #[test]
    fn test_len_increments() {
        let mut t = Treap::new();
        for i in 0..5 {
            t.insert(i);
            assert_eq!(t.len(), (i + 1) as usize);
        }
    }

    #[test]
    fn test_len_no_dup() {
        let mut t = Treap::new();
        t.insert(1);
        t.insert(1);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_delete_empty_false() {
        let mut t: Treap<i32> = Treap::new();
        assert!(!t.delete(&5));
    }

    #[test]
    fn test_delete_found() {
        let mut t = make_treap(&[1, 2, 3]);
        assert!(t.delete(&2));
        assert!(!t.contains(&2));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut t = make_treap(&[1, 2, 3]);
        assert!(!t.delete(&99));
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn test_delete_inorder_sorted() {
        let mut t = make_treap(&[5, 3, 7, 1, 4, 6, 8]);
        t.delete(&5);
        assert_eq!(t.inorder(), vec![&1, &3, &4, &6, &7, &8]);
    }

    #[test]
    fn test_split_basic() {
        let t = make_treap(&[1, 2, 3, 4, 5]);
        let (left, right) = t.split(&3);
        assert_eq!(left.inorder(), vec![&1, &2, &3]);
        assert_eq!(right.inorder(), vec![&4, &5]);
    }

    #[test]
    fn test_split_empty() {
        let t: Treap<i32> = Treap::new();
        let (left, right) = t.split(&5);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn test_split_all_left() {
        let t = make_treap(&[1, 2, 3]);
        let (left, right) = t.split(&10);
        assert_eq!(left.inorder(), vec![&1, &2, &3]);
        assert!(right.is_empty());
    }

    #[test]
    fn test_split_all_right() {
        let t = make_treap(&[5, 10, 15]);
        let (left, right) = t.split(&0);
        assert!(left.is_empty());
        assert_eq!(right.inorder(), vec![&5, &10, &15]);
    }

    #[test]
    fn test_merge_basic() {
        let left = make_treap(&[1, 3]);
        let right = make_treap(&[5, 7]);
        let merged = Treap::merge(left, right);
        assert_eq!(merged.inorder(), vec![&1, &3, &5, &7]);
    }

    #[test]
    fn test_merge_empty_left() {
        let left: Treap<i32> = Treap::new();
        let right = make_treap(&[1, 2]);
        let merged = Treap::merge(left, right);
        assert_eq!(merged.inorder(), vec![&1, &2]);
    }

    #[test]
    fn test_merge_empty_right() {
        let left = make_treap(&[1, 2]);
        let right: Treap<i32> = Treap::new();
        let merged = Treap::merge(left, right);
        assert_eq!(merged.inorder(), vec![&1, &2]);
    }

    #[test]
    fn test_union_basic() {
        let t1 = make_treap(&[1, 3, 5]);
        let t2 = make_treap(&[2, 4, 6]);
        let u = t1.union(t2);
        assert_eq!(u.inorder(), vec![&1, &2, &3, &4, &5, &6]);
    }

    #[test]
    fn test_union_with_overlap() {
        let t1 = make_treap(&[1, 2, 3]);
        let t2 = make_treap(&[2, 3, 4]);
        let u = t1.union(t2);
        assert_eq!(u.inorder(), vec![&1, &2, &3, &4]);
    }

    #[test]
    fn test_intersection_basic() {
        let t1 = make_treap(&[1, 2, 3, 4]);
        let t2 = make_treap(&[2, 4, 6]);
        let i = t1.intersection(&t2);
        assert_eq!(i.inorder(), vec![&2, &4]);
    }

    #[test]
    fn test_intersection_no_overlap() {
        let t1 = make_treap(&[1, 3, 5]);
        let t2 = make_treap(&[2, 4, 6]);
        let i = t1.intersection(&t2);
        assert!(i.is_empty());
    }

    #[test]
    fn test_rank_basic() {
        let t = make_treap(&[1, 2, 3, 4, 5]);
        assert_eq!(t.rank(&3), 2);
        assert_eq!(t.rank(&1), 0);
        assert_eq!(t.rank(&5), 4);
    }

    #[test]
    fn test_select_basic() {
        let t = make_treap(&[5, 3, 7, 1]);
        assert_eq!(t.select(0), Some(&1));
        assert_eq!(t.select(3), Some(&7));
    }

    #[test]
    fn test_select_out_of_range() {
        let t = make_treap(&[1, 2, 3]);
        assert_eq!(t.select(100), None);
    }

    #[test]
    fn test_rank_select_roundtrip() {
        let keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let t = make_treap(&keys);
        for &k in &keys {
            let r = t.rank(&k);
            assert_eq!(t.select(r), Some(&k));
        }
    }

    #[test]
    fn test_mass_insert_inorder() {
        let mut keys: Vec<i32> = (1..=100).collect();
        // Insert in a non-sequential order using a simple permutation
        keys.sort_by_key(|&k| (k * 37 + 13) % 100);
        let t = make_treap(&keys);
        let expected: Vec<i32> = (1..=100).collect();
        let got: Vec<i32> = t.inorder().into_iter().copied().collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_mass_split_merge() {
        let t = make_treap(&(1..=50).collect::<Vec<i32>>());
        let (left, right) = t.split(&25);
        let merged = Treap::merge(left, right);
        let expected: Vec<i32> = (1..=50).collect();
        let got: Vec<i32> = merged.inorder().into_iter().copied().collect();
        assert_eq!(got, expected);
    }
}
