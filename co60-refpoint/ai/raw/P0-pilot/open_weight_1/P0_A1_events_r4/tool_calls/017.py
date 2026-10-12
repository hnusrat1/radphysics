import json

# Check the output
output = json.loads(json_str)
print(f"Number of events: {len(output['events'])}")
print(f"Method length: {len(output['method'])} chars")
print(f"Method: {output['method']}")

# Verify required fields
required = ['process', 'e_in_MeV', 'e_out_MeV', 'cos_scatter', 't_e_MeV', 'cos_electron', 't_positron_MeV']
for i, evt in enumerate(output['events']):
    for key in required:
        if key not in evt:
            print(f"Event {i} missing {key}")
    # Check types
    if evt['process'] not in ('compton', 'photoelectric', 'pair', 'rayleigh'):
        print(f"Event {i} invalid process: {evt['process']}")
    if not isinstance(evt['e_in_MeV'], (int, float)):
        print(f"Event {i} e_in not number")
    if evt['e_out_MeV'] is not None and not isinstance(evt['e_out_MeV'], (int, float)):
        print(f"Event {i} e_out not number or null")
    if evt['cos_scatter'] is not None and not isinstance(evt['cos_scatter'], (int, float)):
        print(f"Event {i} cos_scatter not number or null")
    if not isinstance(evt['t_e_MeV'], (int, float)):
        print(f"Event {i} t_e not number")
    if evt['cos_electron'] is not None and not isinstance(evt['cos_electron'], (int, float)):
        print(f"Event {i} cos_electron not number or null")
    if evt['t_positron_MeV'] is not None and not isinstance(evt['t_positron_MeV'], (int, float)):
        print(f"Event {i} t_positron not number or null")

print("Validation complete.")

# Print first 3 events to verify
for evt in output['events'][:3]:
    print(json.dumps(evt))