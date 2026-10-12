# Generate the final 200 events and output as JSON-like structure
import json

# Re-use the events we already generated (they are in 'events' variable from the previous cell)
# But we need to ensure we have exactly 200
print(f"Number of events: {len(events)}")

# Check types
for e in events:
    assert isinstance(e['e_in_MeV'], (int, float))
    assert e['e_out_MeV'] is None or isinstance(e['e_out_MeV'], (int, float))
    assert e['cos_scatter'] is None or isinstance(e['cos_scatter'], (int, float))
    assert isinstance(e['t_e_MeV'], (int, float))
    assert e['cos_electron'] is None or isinstance(e['cos_electron'], (int, float))
    assert e['t_positron_MeV'] is None or isinstance(e['t_positron_MeV'], (int, float))
    assert e['process'] in ('compton', 'photoelectric', 'pair', 'rayleigh')

print("All events pass type checks.")

# Prepare JSON output
method_text = (
    "The fluence spectrum at the scoring volume was computed by combining the attenuated primary (2.0 MeV) "
    "with a single-Compton-scatter spectrum obtained via Monte Carlo integration over the phantom volume using "
    "Klein-Nishina cross sections and attenuation. The spectrum was scaled to approximate multiple scatter contributions. "
    "Incident photon energies were sampled from this spectrum weighted by total interaction cross section. "
    "For each energy, the interaction process was sampled proportional to its cross section, and the outcome "
    "(scattering angle, electron energy, etc.) was sampled from the relevant physical distributions."
)

output = {
    "method": method_text,
    "events": events
}

# Print as JSON
print(json.dumps(output, indent=2)[:500])  # preview truncated