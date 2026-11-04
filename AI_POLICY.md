# AI Policy

LLM use is allowed.
But all contributions shall involve a human author who understands and is responsible for the changes.
Only a human can be an author.
Maintainers may close PRs/issues that appear not to meet these standards.

**Code Contributions:**
Contributions shall not be fully autonomous.
The author shall create pull requests with a description they wrote and shall self-review any LLM-generated content before requesting review.
We recommend opening the pull request as a draft until self-review is complete.
Commits whose content was predominantly produced by an LLM shall disclose this, as AGENTS.md specifies.
LLMs shall be configured to read AGENTS.md.

**Documentation:**
Public documentation and public docstrings shall be *authored* by a human.
Authors may use an LLM as an editor for translation, paraphrasing, rewording, style, grammar, spelling, formatting, suggesting changes, and review.
The author shall review documentation for clarity and correctness as a final pass.
Authors shall verify all citations, as LLMs often hallucinate specifics, such as page numbers.

**Conversations:**
All user-to-user communication, such as PR descriptions, issues, comments, code reviews, and feedback, shall be written by a human for a human.
Users may use an LLM as an editor (same rules as documentation).
However, users may include code, quote boxes, or quote marks with the output from LLMs.
LLM text shall be supplemental, meaning the quoted output can be skipped.
For example: "an LLM reported 'X', so I checked Y, and I think Z."

**Review:**
Reviewers may use LLMs as part of a human-driven review.
Reviewers shall verify LLM findings before reporting them, and shall summarize them in comments they wrote.
This keeps discussions focused and avoids false positives.
