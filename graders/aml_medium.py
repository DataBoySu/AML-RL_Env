def grade(trajectory) -> float:
    final_action = trajectory[-1].action.action
    
    if final_action.action_type != "submit_decision":
        return 0.01
        
    if final_action.decision == "FRAUD":
        evidence = final_action.evidence_links
        # Check if they caught the 3 Smurfs
        caught = sum(1 for smurf in ["ACC-9011", "ACC-9012", "ACC-9013"] if smurf in evidence)
        if caught == 3:
            return 1.0
        elif caught > 0:
            return 0.4 + (caught * 0.1) # Partial credit
            
    return 0.1 # Failed to catch the structuring