# AppWorld notice

The AppWorld runtime and benchmark are developed by the AppWorld authors and
distributed under Apache License 2.0, with an additional rule for the protected
portion: public redistribution of protected content or its derivatives must be
in an encrypted format.

CANOPY's `recipe/appworld/env_server/prompts.py` is based on AppWorld's public
legacy ReAct code-agent instructions at revision
`ba33afb327152803956fdc16f2c3b94a88377453`:

<https://github.com/StonyBrookNLP/appworld/blob/ba33afb327152803956fdc16f2c3b94a88377453/experiments/prompts/react_code_agent/_legacy_instructions.txt>

CANOPY removes the upstream template's final `Task: {{ input_str }}` line
because its environment service sends the actual task as a separate user
message, and adds an optional local file override. The repository does not
distribute protected task/API data, databases, evaluator content, split
Parquet files, or protected derivatives. Users must obtain the remaining
AppWorld materials through its official distribution and comply with its
terms. See:

<https://github.com/StonyBrookNLP/appworld#-license>

The full Apache License 2.0 text is retained in `LICENSE` beside this notice.
