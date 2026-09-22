"""HTTP adapter around the existing agent.

This package translates between HTTP and the domain objects in `app`. It may
load a dataset, build an Agent, ask it a question and convert the result to
JSON. It may not analyse anything: no pandas, no tools, no validator, no
executor, no planner. A test in tests/test_api.py enforces that.
"""
