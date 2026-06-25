from transcript import load_transcript
from rag import build_rag_chain


def main():
    print("-" * 60)
    print("YouTube RAG Assistant")
    print("-" * 60)

    video_url = input("\nEnter YouTube URL:\n> ")
    
    try:
        transcript = load_transcript(video_url)
        print("\nTranscript Loaded Successfully.")

        chain = build_rag_chain(transcript)

        print("\nYou can now ask questions.")
        print("Type 'exit' to quit.\n")

        while True:
            question = input("> ")

            if question.lower() == "exit":
                break

            answer = chain.invoke(question)
            print("\nAnswer:\n")

            print(answer)
            print()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()