"""
Generates a small labeled movie/product review dataset for demo training.
NOTE: This is a synthetic template-based dataset for demonstration purposes.
For production-grade accuracy, swap this out with a real dataset like IMDB
(https://ai.stanford.edu/~amaas/data/sentiment/) or Amazon reviews.
"""
import csv
import random

random.seed(42)

positive_templates = [
    "This {noun} was absolutely {pos_adj}, I loved every moment of it.",
    "One of the best {noun}s I have experienced, truly {pos_adj}.",
    "I was so impressed, the {noun} exceeded my expectations completely.",
    "{pos_adj} from start to finish, highly recommend this {noun}.",
    "The {noun} made me so happy, everything about it felt {pos_adj}.",
    "Great {noun}, well worth the time and money, very {pos_adj}.",
    "I can't stop thinking about how {pos_adj} this {noun} was.",
    "Such a {pos_adj} experience, I will definitely come back for more.",
    "The quality of this {noun} is outstanding and genuinely {pos_adj}.",
    "Everyone should try this {noun}, it is simply {pos_adj}.",
    "What a {pos_adj} surprise, this {noun} blew me away.",
    "I felt so satisfied after the {noun}, it was {pos_adj} in every way.",
]

negative_templates = [
    "This {noun} was absolutely {neg_adj}, I regret spending time on it.",
    "One of the worst {noun}s I have experienced, truly {neg_adj}.",
    "I was so disappointed, the {noun} fell short of expectations badly.",
    "{neg_adj} from start to finish, would not recommend this {noun}.",
    "The {noun} made me so frustrated, everything about it felt {neg_adj}.",
    "Terrible {noun}, not worth the time or money, very {neg_adj}.",
    "I can't believe how {neg_adj} this {noun} turned out to be.",
    "Such a {neg_adj} experience, I will never come back again.",
    "The quality of this {noun} is poor and genuinely {neg_adj}.",
    "No one should try this {noun}, it is simply {neg_adj}.",
    "What a {neg_adj} letdown, this {noun} was a waste of time.",
    "I felt so annoyed after the {noun}, it was {neg_adj} in every way.",
]

nouns = ["movie", "product", "service", "restaurant", "book", "app", "show", "game", "hotel", "gadget"]
pos_adj = ["amazing", "fantastic", "wonderful", "brilliant", "delightful", "excellent", "superb", "incredible"]
neg_adj = ["awful", "horrible", "disappointing", "dreadful", "mediocre", "boring", "frustrating", "poor"]

rows = []
for _ in range(220):
    t = random.choice(positive_templates)
    text = t.format(noun=random.choice(nouns), pos_adj=random.choice(pos_adj))
    rows.append((text, "positive"))

for _ in range(220):
    t = random.choice(negative_templates)
    text = t.format(noun=random.choice(nouns), neg_adj=random.choice(neg_adj))
    rows.append((text, "negative"))

random.shuffle(rows)

with open("data/reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> data/reviews.csv")
