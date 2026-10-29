import json

with open('final.json', 'r') as f:
    content = f.read()

header = '{"method":"Monte Carlo simulation using analytical cross-section models (Klein-Nishina, Sauter approx, linear pair-production model) for liquid water. Energy sampling weighted by primary attenuation at 5 cm depth.","events":['

# The file already contains the full JSON including header?
# Let's check the first char
if content.startswith('{'):
    print("File already has header.")
    # Just print it
    sys.stdout.write(content)
else:
    # Re-add header if needed (though we saved it before)
    pass
