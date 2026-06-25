from transcript import load_transcript
from rag import build_rag_chain


def main():

    print("-" * 60)
    print("YouTube RAG Assistant")
    print("-" * 60)

    while True:
        video_url = input("\nEnter YouTube URL:\n> ").strip()
        try:
            print("\nDownloading transcript...")
            transcript = load_transcript(video_url)

            print("Building vector database...")
            chain = build_rag_chain(transcript)

            print("\nReady!")
            print("Ask questions about this video.")
            print("Type 'exit' when you're done.\n")

            while True:
                question = input("> ").strip()

                if question.lower() == "exit":
                    break

                answer = chain.invoke(question)

                print("-"*60)
                print()
                print("\nAnswer:\n")
                print(answer)
                print()
                print("-"*60)
                print()

        except Exception as e:
            print(f"\nError: {e}")

        again = input("\nAnalyze another YouTube video? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nThanks for using YouTube RAG Assistant!")


if __name__ == "__main__":
    main()