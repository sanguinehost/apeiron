# A Cell

```rust
extern crate apeiron;
use apeiron::{Cell, Sigma};

let x = Cell::<i64>::of(3, 2);
let y = Cell::<i64>::of(1, 5);
let p = x.mul(Sigma::Hyperbolic, y);
assert_eq!((p.a, p.b), (3 + 2 * 5, 3 * 5 + 2 * 1));
```
