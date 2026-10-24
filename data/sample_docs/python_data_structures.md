# Python Data Structures - Quick Reference

## Lists
A list is an ordered, mutable collection written with square brackets, e.g. `[1, 2, 3]`.
Use `append()` to add one item, `extend()` to add many, and `pop()` to remove the last item.
Accessing an item by index is O(1), but searching for a value with `in` is O(n).

## Tuples
A tuple is an ordered, immutable collection written with parentheses, e.g. `(10, 20)`.
Because tuples cannot change, they can be used as dictionary keys and are slightly faster than lists.

## Dictionaries
A dictionary stores key-value pairs, e.g. `{"name": "Asha", "age": 21}`.
Lookups, inserts and deletes are O(1) on average because dictionaries are hash tables.
Since Python 3.7, dictionaries keep the order in which keys were inserted.

## Sets
A set is an unordered collection of unique items, e.g. `{1, 2, 3}`.
Sets are ideal for removing duplicates and for fast membership tests.
They support union (`|`), intersection (`&`) and difference (`-`).

## Choosing the right structure
Use a list for ordered data that changes, a tuple for fixed records, a dictionary for
fast lookup by key, and a set when only uniqueness matters.
