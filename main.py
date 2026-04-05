from src.predict import predict_sentiment


def run_cli() -> None:
    print("Sentiment Analysis CLI")
    print("Type 'exit' to quit.")

    while True:
        text = input("\nEnter text: ").strip()
        if text.lower() == "exit":
            print("Goodbye.")
            break
        if not text:
            print("Please enter non-empty text.")
            continue

        result = predict_sentiment(text)
        print(f"Label      : {result['label']}")
        print(f"Confidence : {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    run_cli()
