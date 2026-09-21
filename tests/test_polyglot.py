from src.polyglot.languages import select_language


def test_default_algorithm_problem_uses_python():
    assert select_language("build a prototype for sorting data").name == "python"


def test_memory_problem_prefers_rust():
    assert select_language("design a low level memory allocator").name == "rust"


def test_performance_problem_prefers_cpp():
    assert select_language("find a very fast algorithm").name == "cpp"


def test_concurrency_problem_prefers_go():
    assert select_language("build a concurrent server").name == "go"
