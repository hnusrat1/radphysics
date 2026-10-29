
method=("Analog Monte Carlo written in numpy (no transport package). Point source 100 cm from surface; directions sampled uniformly in solid angle "
        "inside the 10x10 cm^2 pyramid. 2 MeV photons traced through the 30 cm water cube with exponential free paths and water cross sections "
        "(Klein-Nishina Compton, Rayleigh, photoelectric, pair). Compton by Klein-Nishina rejection; annihilation photons tracked. "
        "All interactions inside the 1 cm radius, 0.2 cm slab at 10 cm depth were recorded from ~1.2e8 histories (~44,600 events); 200 drawn at random without replacement.")
print(len(method.split()))
out={"method":method,"events":events}
print(json.dumps(out)[:200])
json.dumps(out)
