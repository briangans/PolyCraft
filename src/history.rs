//! Undo / redo history for the document.
//!
//! `HistoryStack` is pure: it knows nothing about meshes, `wgpu`, or `winit`. It is generic over the
//! snapshot type `T` - whatever the caller wants versioned - so the stack logic can be unit-tested on
//! its own with `T = i32`.
//!
//! Snapshots are taken explicitly *around* a user interaction rather than on every mutation:
//! [`HistoryStack::begin`] captures the state before the interaction starts and [`HistoryStack::commit`]
//! records it when the interaction ends (a gizmo drag, an extrude). That collapses a whole drag -
//! hundreds of `translate_vertices` calls - into a single undo step, and keeps the number of snapshots
//! proportional to the number of actions the user took rather than the number of frames drawn.
//!
//! The semantics match Maya: recording a new action drops the redo branch, an action that changed
//! nothing is never recorded, and the depth is bounded so memory cannot grow without limit.

/// One undoable step: the human-readable action name, plus the document state from *before* it ran.
type Step<T> = (String, T);

#[derive(Clone, Debug)]
pub struct HistoryStack<T> {
    /// Steps the user can step back through, oldest first. Each holds the state to return to.
    undo: Vec<Step<T>>,
    /// Steps that were undone and can be re-applied, most recently undone last.
    redo: Vec<Step<T>>,
    /// The step currently being recorded, between `begin` and `commit`/`cancel`.
    pending: Option<Step<T>>,
    /// Oldest steps beyond this depth are dropped to bound memory.
    limit: usize,
}

impl<T: Clone + PartialEq> HistoryStack<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            pending: None,
            limit: limit.max(1),
        }
    }

    /// Starts recording an action, capturing `current` as the state `undo` will return to.
    ///
    /// Calling this while an action is already open keeps the outermost state and label, so a
    /// repeated or nested press can never shrink the step down to only part of the interaction.
    pub fn begin(&mut self, label: &str, current: &T) {
        if self.pending.is_none() {
            self.pending = Some((label.to_owned(), current.clone()));
        }
    }

    /// Closes the open action, recording it only if `current` differs from the state captured by
    /// `begin`. Clicking a gizmo axis and releasing without moving therefore creates no undo step.
    /// Recording a step drops the redo branch, because the timeline has forked. Returns true when a
    /// step was recorded.
    pub fn commit(&mut self, current: &T) -> bool {
        let Some((label, before)) = self.pending.take() else {
            return false;
        };
        if before == *current {
            return false;
        }

        self.redo.clear();
        if self.undo.len() >= self.limit {
            self.undo.remove(0);
        }
        self.undo.push((label, before));
        true
    }

    /// Throws away the open action without recording it, as Escape does for an in-progress drag.
    pub fn cancel(&mut self) {
        self.pending = None;
    }

    /// Steps back one action: `current` is the state being undone, and is kept so `redo` can put it
    /// back. Returns the state from before the action, or `None` when there is nothing to undo.
    pub fn undo(&mut self, current: &T) -> Option<T> {
        let (label, before) = self.undo.pop()?;
        self.redo.push((label, current.clone()));
        Some(before)
    }

    /// Re-applies the most recently undone action. `current` is the state being replaced, which goes
    /// back onto the undo branch so the user can step back again.
    pub fn redo(&mut self, current: &T) -> Option<T> {
        let (label, after) = self.redo.pop()?;
        self.undo.push((label, current.clone()));
        Some(after)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Name of the action `undo` would revert, for "Undo - Move" style UI labels.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|(label, _)| label.as_str())
    }

    /// Name of the action `redo` would re-apply.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|(label, _)| label.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the stack with plain integers, so the stack logic is tested without any document types.
    fn step_history() -> HistoryStack<i32> {
        HistoryStack::new(4)
    }

    /// Records an action that moves the document from `state` to `state + 1`.
    fn record(stack: &mut HistoryStack<i32>, label: &str, state: i32) {
        stack.begin(label, &state);
        stack.commit(&(state + 1));
    }

    #[test]
    fn starts_empty() {
        let stack = step_history();
        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
        assert_eq!(stack.undo_label(), None);
        assert_eq!(stack.redo_label(), None);
    }

    #[test]
    fn recorded_action_can_be_undone_and_redone() {
        let mut stack = step_history();
        record(&mut stack, "Move", 0);
        assert_eq!(stack.undo_label(), Some("Move"));

        assert_eq!(stack.undo(&1), Some(0), "undo returns the before state");
        assert!(!stack.can_undo());
        assert!(stack.can_redo());
        assert_eq!(stack.redo_label(), Some("Move"));

        assert_eq!(stack.redo(&0), Some(1), "redo re-applies the action");
        assert!(stack.can_undo());
        assert!(!stack.can_redo());
    }

    #[test]
    fn nothing_is_undoable_until_the_action_is_committed() {
        let mut stack = step_history();
        stack.begin("Move", &0);
        assert!(
            !stack.can_undo(),
            "an uncommitted action must not appear in the undo stack"
        );

        stack.cancel();
        assert!(!stack.can_undo());
    }

    #[test]
    fn an_action_that_changed_nothing_is_not_recorded() {
        let mut stack = step_history();
        stack.begin("Move", &3);

        assert!(
            !stack.commit(&3),
            "clicking a gizmo axis without dragging records nothing"
        );
        assert!(!stack.can_undo());
    }

    #[test]
    fn committing_with_no_open_action_does_nothing() {
        let mut stack = step_history();
        record(&mut stack, "Move", 0);

        assert!(!stack.commit(&1), "a stray commit cannot invent a step");
        assert_eq!(stack.undo_label(), Some("Move"));
    }

    #[test]
    fn a_second_begin_keeps_the_outermost_state() {
        let mut stack = step_history();
        stack.begin("Move", &0);
        stack.begin("Move", &99);

        assert!(stack.commit(&1));
        assert_eq!(
            stack.undo(&1),
            Some(0),
            "a nested begin must not shrink the step being undone"
        );
    }

    #[test]
    fn a_new_action_drops_the_redo_branch() {
        let mut stack = step_history();
        record(&mut stack, "Move", 0);
        stack.undo(&1);
        assert!(stack.can_redo());

        record(&mut stack, "Extrude", 1);
        assert!(
            !stack.can_redo(),
            "editing after an undo forks the timeline and discards redo"
        );
        assert_eq!(stack.undo_label(), Some("Extrude"));
    }

    #[test]
    fn steps_undo_and_redo_in_reverse_order() {
        let mut stack = step_history();
        record(&mut stack, "First", 0);
        record(&mut stack, "Second", 1);
        record(&mut stack, "Third", 2);

        assert_eq!(stack.undo(&3), Some(2));
        assert_eq!(stack.undo(&2), Some(1));
        assert_eq!(stack.undo(&1), Some(0));
        assert_eq!(stack.undo(&0), None, "history is exhausted, not rewound");

        assert_eq!(stack.redo(&0), Some(1));
        assert_eq!(stack.redo(&1), Some(2));
        assert_eq!(stack.redo(&2), Some(3));
        assert_eq!(stack.redo(&3), None);
    }

    #[test]
    fn the_oldest_step_is_dropped_beyond_the_depth_limit() {
        let mut stack = HistoryStack::new(2);
        record(&mut stack, "One", 0);
        record(&mut stack, "Two", 1);
        record(&mut stack, "Three", 2);

        assert_eq!(stack.undo_label(), Some("Three"));
        assert_eq!(stack.undo(&3), Some(2));
        assert_eq!(stack.undo(&2), Some(1));
        assert_eq!(
            stack.undo(&1),
            None,
            "'One' fell off the 2-deep stack, so the oldest change is unrecoverable"
        );
    }

    #[test]
    fn a_limit_of_zero_still_holds_one_step() {
        let mut stack = HistoryStack::new(0);
        record(&mut stack, "Move", 0);
        assert_eq!(stack.undo(&1), Some(0));
    }

    #[test]
    fn redo_after_undo_keeps_the_step_available_again() {
        let mut stack = step_history();
        record(&mut stack, "Move", 5);

        stack.undo(&6);
        stack.redo(&5);
        assert_eq!(
            stack.undo(&6),
            Some(5),
            "stepping back after a redo must still return to the original state"
        );
    }
}
