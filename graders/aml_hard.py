def grade(trajectory) -> float:
    final_action = trajectory[-1].action.action
    
    if final_action.action_type != "submit_decision":
        return 0.01
        
    if final_action.decision == "FRAUD":
        evidence = final_action.evidence_links
        
        # Massive penalty if they fell for the Al-Qaeda false flag
        if "ACC-9026" in evidence or "ENT-9026" in evidence:
            return 0.05 
            
        # Check if they found the hidden circular loop
        caught = sum(1 for link in ["ACC-9021", "ACC-9022", "ACC-9023"] if link in evidence)
        if caught == 3:
            return 1.0
        elif caught > 0:
            return 0.5 # Found part of the loop
            
    return 0.1