from ai_scientist.llm import Model

def test_local_model():
    try:
        model = Model("llama3.2:1b")
        response = model.get_response("Hello, how are you?")
        print("Response:", response)
        return True
    except Exception as e:
        print("Error:", str(e))
        return False

if __name__ == "__main__":
    test_local_model()
