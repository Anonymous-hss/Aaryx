from rag import portfolio_search

print("Testing direct RAG search...")
results = portfolio_search.invoke("What projects has Harshal built?")
print("SEARCH RESULTS:")
print(results)
