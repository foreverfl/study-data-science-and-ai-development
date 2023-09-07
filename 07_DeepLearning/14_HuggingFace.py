from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results1 = classifier("We are very happy.")
print('result1:', results1)
results2 = classifier(["We are very happy.", "We hope you don't hate it."])
print('result2:', results2)
