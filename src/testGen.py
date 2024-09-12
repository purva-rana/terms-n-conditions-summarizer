import random

def generate_example(id, label):
    benign_templates = [
        "Our customer support team is available to help you with any questions about our services.",
        "You can easily review and update your personal information through your account settings.",
        # Add more benign templates here
    ]

    malicious_templates = [
        "We may use your personal data to create targeted marketing profiles for our partners.",
        "We reserve the right to alter or remove features without prior notice to users.",
        # Add more malicious templates here
    ]

    if label == "Benign":
        text = random.choice(benign_templates)
    elif label == "Malicious":
        text = random.choice(malicious_templates)
    
    return f"{id},{text},{label}\n"

# Example of generating 100 examples
with open('examples.csv', 'w') as file:
    for i in range(1, 101):
        label = "Benign" if i % 2 == 0 else "Malicious"
        file.write(generate_example(i, label))
