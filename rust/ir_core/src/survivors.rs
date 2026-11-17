//! Incremental resolution of passthrough instruction views.
//!
//! A view renders the nodes on its chain that are not superseded: every node
//! without a replace key, and for each replace key only the node whose version
//! the view lists as latest. Walking the chain to find them is linear in its
//! length, and the chain grows with every graphic-state operator on the page,
//! so rendering one view per path element is quadratic on figures drawn as
//! tens of thousands of separately coloured paths.
//!
//! Instead, each node carries the survivors of its own chain, derived from its
//! parent's: a replace-keyed node drops the previous node with its key and
//! adds itself. The list is persistent, so a node shares everything older than
//! the entry it removed. A view whose latest versions are the ones its chain
//! implies (true of every view the parser makes) renders straight from its
//! tail's list; any other view falls back to the walk.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::{History, HistoryNode};

/// Survivors of a chain, newest first; `node` is a local history index.
pub(crate) struct Kept {
    node: usize,
    older: Option<Arc<Kept>>,
}

impl Drop for Kept {
    fn drop(&mut self) {
        // Iterative: a long unshared tail would otherwise recurse once per entry.
        let mut older = self.older.take();
        while let Some(cell) = older {
            match Arc::try_unwrap(cell) {
                Ok(mut cell) => older = cell.older.take(),
                Err(_) => break,
            }
        }
    }
}

#[derive(Clone)]
pub(crate) enum Resolved {
    Kept(Option<Arc<Kept>>),
    /// A chain whose versions for some replace key do not increase, which
    /// could give two of its nodes the same key and version: only the walk
    /// reproduces how such a chain renders.
    Walk,
}

pub(crate) type ResolvedSlot = OnceLock<Resolved>;

impl History {
    /// The nodes a view with tail `tail` and these latest versions renders,
    /// oldest first, or None when only a full walk of the chain can tell.
    pub(crate) fn survivors(
        &self,
        tail: usize,
        latest: &HashMap<&str, u64>,
    ) -> Result<Option<Vec<&HistoryNode>>, super::AccessError> {
        let tail = self.index(tail)?;
        let Some(Resolved::Kept(list)) = self.resolve(tail) else {
            return Ok(None);
        };
        let mut nodes = Vec::new();
        let mut at = list.as_deref();
        while let Some(cell) = at {
            let node = &self.nodes[cell.node];
            if let Some(key) = node.replace_key.as_deref()
                && latest.get(key).copied() != Some(node.version)
            {
                // The view's latest versions are not the ones its chain implies.
                return Ok(None);
            }
            nodes.push(node);
            at = cell.older.as_deref();
        }
        nodes.reverse();
        Ok(Some(nodes))
    }

    fn resolve(&self, tail: usize) -> Option<Resolved> {
        // History compaction rebuilds the node table; until the slots match it
        // again, nothing here can be trusted.
        if self.resolved.len() != self.nodes.len() {
            return None;
        }
        let mut pending = Vec::new();
        let mut at = Some(tail);
        let mut resolved = Resolved::Kept(None);
        while let Some(i) = at {
            if let Some(known) = self.resolved[i].get() {
                resolved = known.clone();
                break;
            }
            pending.push(i);
            at = self.nodes[i].parent;
        }
        for i in pending.into_iter().rev() {
            let next = match resolved {
                Resolved::Walk => Resolved::Walk,
                Resolved::Kept(list) => self.extend(list, i),
            };
            resolved = self.resolved[i].get_or_init(|| next).clone();
        }
        Some(resolved)
    }

    /// Survivors of node `i`'s chain, given those of its parent's chain.
    fn extend(&self, list: Option<Arc<Kept>>, i: usize) -> Resolved {
        let node = &self.nodes[i];
        let older = match node.replace_key.as_deref() {
            None => list,
            Some(key) => {
                // Copy the entries newer than the one being superseded, then
                // relink them onto what is older than it.
                let mut newer = Vec::new();
                let mut at = list.as_ref();
                let mut rest = None;
                while let Some(cell) = at {
                    let kept = &self.nodes[cell.node];
                    if kept.replace_key.as_deref() == Some(key) {
                        if kept.version >= node.version {
                            return Resolved::Walk;
                        }
                        rest = Some(cell.older.clone());
                        break;
                    }
                    newer.push(cell.node);
                    at = cell.older.as_ref();
                }
                match rest {
                    None => list,
                    Some(mut rebuilt) => {
                        for node in newer.into_iter().rev() {
                            rebuilt = Some(Arc::new(Kept {
                                node,
                                older: rebuilt,
                            }));
                        }
                        rebuilt
                    }
                }
            }
        };
        Resolved::Kept(Some(Arc::new(Kept { node: i, older })))
    }

    pub(crate) fn reset_resolved(&mut self) {
        self.resolved.clear();
        self.resolved.resize_with(self.nodes.len(), OnceLock::new);
    }
}
