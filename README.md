## znn

Simple Tensor library in zig with (eventual) autograd.

## goals

I want to implement a simple, readable and performant Tensor library as a way
of getting more familiar with the inner workings of larger deep learning
frameworks myself.

Starting out it will likely be limited to CPU compute for simplicity but it's
possible that I decide to add generic device support later on if the simplicity
trade off isn't too great.

## motivation

I started work on a similar project a while ago in Rust, but ran into a lot of
issues with the borrow-checker.

As a result, this time revisiting the project, I decided to start out using Zig
and draw up a MVP, without having to fight the borrow-checker, for what an
educational, simple but still relatively performant version of my project would
look like so I later could implement it in Rust.

The reason is that I found it hard to reason about the larger problem space as
well as API decisions while simultaneously dealing with the limitations of the
borrow-checker. The hope is that if I can first form an idea of the overall
project structure, I will have an easier time dealing with the borrow-checker
after.
