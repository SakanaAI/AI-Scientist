import matplotlib.pyplot as plt

# Data for ACL paper attack types
labels = ['White Text', 'Metadata', 'Invisible Chars',
          'Mixed Language', 'Steganographic', 'Contextual Attack']
sizes = [30, 25, 20, 15, 7, 3]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#99ccff']

# Create pie chart
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Add title
plt.title('Distribution of Prompt Injection Attack Types in Synthetic Data', pad=20)

# plt.show()

# Save the plot
plt.savefig('attack_types_distribution.png', bbox_inches='tight', dpi=300)
