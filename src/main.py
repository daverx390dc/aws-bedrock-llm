from qa_chain import get_qa_chain

if __name__ == '__main__':
    print("Loading index...")
    qa_chain = get_qa_chain()
    print("Ready for interaction...")

    while True:
        query = input('\nEnter query: ')
        if query.lower() in ("exit", "quit"):
            break
        answer = qa_chain.invoke(query)
        print(f"Answer: {answer}")
