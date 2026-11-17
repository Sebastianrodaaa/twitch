# Ponytail

Trace the real flow end to end before choosing an implementation. Take the first rung that holds.

1. Does it need to exist? Skip speculative work and say why in one line.
2. Search the codebase for an existing solution. Reuse its types, helpers, and patterns.
3. Use the standard library if it solves the problem.
4. Use the platform feature if it already covers the need.
5. Use an installed dependency if it solves the problem. A few lines rarely justify a new dependency.
6. Make it one line when that stays readable.
7. Write the minimum code that works.

Grep callers before edits and fix root causes at the shared point. Prefer deletion, boring code, the fewest files, and the shortest correct diff. Skip single-implementation interfaces, factories and scaffolding for hypothetical requirements. Mark deliberate ceilings with a `ponytail:` comment describing the upgrade path.

Preserve validation at trust boundaries, error handling that prevents data loss, security, accessibility basics, and all explicit requirements. Leave one runnable check for non-trivial logic. Use no test framework unless CI needs one.
