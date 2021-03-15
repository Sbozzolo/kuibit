# How to contirbute to `kuibit`

## Bug reports

If you find a bug, an un-helpful error message, or an expected result, please
report it. If you can provide precise steps to reproduce the problem, that would
certainly speed-up the resolution of the bug.

You can also join us in the [Telegram channel](https://t.me/kuibit) for help or
discussions.

## New features/improvements

We always welcome new features of general improvements in the code. If you have
some scripts based on ``kuibit`` that implement functionalities useful to
the community, we can work together to port them in the main package. (Your
contributed code should not depend on private thorns or other private codes.)

You want to contribute but don't know what to do? If you want an idea of
possible projects, you can have a look at the [TODO](TODO.md "TODO").

### Unittest

Make sure to test your code with `unittest`. Try different inputs, also the ones
that will lead to errors. Remember, even 100% coverage is not a guarantee of
working code.

### Documentation

## Documentation improvements

If you find some features difficult to use and you feel there is not enough
documentation, please report it! We will be happy to extend the documentation
where it is lacking, or provide more examples. If you want to write the missing
documentation yourself, we will be very excited to accept the contribution.

## Code conventions

We are explicit in naming variables, functions, methods, and classes. Long
descriptive names are preferred over short ones.

We apply `black -l 79` to all the files before committing them.

Modules that work directly with Cactus/Carpet objects have start with `cactus`
in their name (e.g., `cactus_grid_functions` vs `grid_data`).

In the spirit of [https://conventionalcomments.org/](conventional comments),
we write our TODO comments in the following way:

```python
# TODO (TYPE): Subject
#
# Description of the problem.
```

`TYPE` has to be capitalized and has to be one of `FEATURE`, `FUTURE`,
`REFACTORING`. The `FUTURE` keyword means that the TODO item
depends on something else being accomplished, for example, an external
dependency implementing some feature.

NumPy is spelled like that: NumPy (not numpy, or Numpy). Similarly SciPy.

## Contributing process

We follow the [https://guides.github.com/introduction/flow/](GitHub flow) for
development and deployment of `kuibit`. If you want to contribute, the steps
are (make sure to read below this list!):

0. For contributions longer than a few lines, announce on GitHub or Telegram
   that you are working on a given feature.
1. Fork `kubit` on GitHub.
2. Create a branch. If your contribution is a bugfix, an improvement in
   documentation, or a small feature, you should branch off `master`. If it is a
   significant improvement, you should branch off `experimental`.
3. Work on your contribution, making sure that your commits are atomic.
4. Add tests, all the test in `kuibit` must pass.
5. Add documentation. This must include comments in the code, docstrings, user
   documentation, and possibly examples.
6. Run `black -l 79`.
7. Optionally, run other static analysis tools like `pylint`, and/or `flake8` to
   improve your code.
8. Open a pull request.
9. Your code will be reviewed by at least one maintainer, who may request some
   additional changes. At the end of the review, the contribution will be
   merged.

This looks like an intimidating list, and if you have never done anything like
this, it may feel overwhelming. These contribution rules are required to ensure
that `kuibit` will have a long life and that people will want to use it and
extend it. Don't let this discourage you! We are here to help, and if you have
the desire to contribute but not the technical knowledge about the
aforementioned steps, we will guide you through the process and teach you what
you need to know. As long as you are in contact with the maintainers on Telegram
or on GitHub, we can work together to get your contribution in `kuibit`. (For
example, this could involve the maintainers fixing the non-passing tests, or
adding docstrings.)

## Notes

### Versions

- We use semantic versioning MAJOR.MINOR.PATCH

  MAJOR is when there is a breaking change
  MINOR is when there are new features. When this is updated, we reset patch
  to zero.
  PATCH is a bug fix or backwards compatible changes.

When releasing a new version, these are the steps:

- `poetry version <type_of_bump>`, with type of bump in the table in [poetry
  documentation](https://python-poetry.org/docs/cli/#version) (`patch`,
  `minor`, `major`, `prepatch`, `preminor`, `premajor`, `prerelease`).
- Edit `kuibit/__init__.py`.
- Edit `tests/test_kuibit.py`.
- Edit `docs/conf.py`
