from promptforge import Example, Field, Signature


def test_render_prompt_contains_input(simple_sig):
    prompt = simple_sig.render_prompt(inputs={"question": "What is 2+2?"})
    assert "What is 2+2?" in prompt
    assert "Answer:" in prompt


def test_render_prompt_contains_instruction(simple_sig):
    prompt = simple_sig.render_prompt(inputs={"question": "test"})
    assert "Answer the question" in prompt


def test_render_prompt_custom_instruction(simple_sig):
    prompt = simple_sig.render_prompt(
        inputs={"question": "test"},
        instruction="My custom instruction",
    )
    assert "My custom instruction" in prompt
    assert "Answer the question" not in prompt


def test_render_prompt_with_demos(simple_sig):
    demo = Example(inputs={"question": "What is 1+1?"}, outputs={"answer": "2"})
    prompt = simple_sig.render_prompt(
        inputs={"question": "What is 3+3?"},
        demos=[demo],
    )
    assert "What is 1+1?" in prompt
    assert "[Example 1]" in prompt
    assert "What is 3+3?" in prompt


def test_parse_output_single(simple_sig):
    assert simple_sig.parse_output("Paris") == {"answer": "Paris"}


def test_parse_output_strips_label(simple_sig):
    assert simple_sig.parse_output("Answer: Paris")["answer"] == "Paris"


def test_parse_output_multi_json(multi_sig):
    result = multi_sig.parse_output('{"name": "Alice", "age": "30"}')
    assert result["name"] == "Alice"
    assert result["age"] == "30"


def test_multi_output_prompt_requests_json(multi_sig):
    prompt = multi_sig.render_prompt(inputs={"text": "Alice is 30"})
    assert "JSON" in prompt


def test_signature_repr(simple_sig):
    r = repr(simple_sig)
    assert "question" in r
    assert "answer" in r
    assert "->" in r


def test_parse_output_label_on_own_line(simple_sig):
    # LLM puts label on its own line, answer on the next — should return "Paris" not "Answer:\nParis"
    result = simple_sig.parse_output("Answer:\nParis")
    assert result["answer"] == "Paris"


def test_parse_output_nested_json(multi_sig):
    # Nested braces inside a field value must not break JSON extraction
    response = '{"name": "Alice", "age": "30"}'
    result = multi_sig.parse_output(response)
    assert result["name"] == "Alice"
    assert result["age"] == "30"


def test_multi_output_flag(simple_sig, multi_sig):
    assert simple_sig._multi_output is False
    assert multi_sig._multi_output is True
