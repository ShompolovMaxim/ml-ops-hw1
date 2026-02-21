## Code Style
- Use PEP8 for Python
- Naming: snake_case

## Plan & Act
- At the first step (plan) you need to read all project files, analyze project structure and develop detailed featue implementation plan and write it into FEATURE_PLAN.md file. Plan must include (in the same order as listed further): implementation steps, testing, linting, documentation updates. On this step you should not implement anything
- At the second step (act) you should perform all actions step by step according to the plan that you wave written into the FEATURE_PLAN.md file. All actions listed in the FEATURE_PLAN.md file should be performed in the same order as they are in the plan. No action can be skipped. The result of each step should be reported to the chat.

## Commands
- Run tests: `py -3.11 -m pytest --verbose`
- Lint code: `py -3.11 -m ruff check .`

## Documentation
- Any API change must update API.md
- All changes must be documented in CHANGELOG.md (detailed)

## Pre-Completion Checks
- Run linter and tests after finishing feature developement
- If linter or tests report issues, you should fix the code and rerun linter and tests