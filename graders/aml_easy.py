def grade(trajectory) -> float:
    # OpenEnv graders usually receive the full trajectory of (observation, action, reward, state)
    # We just need to look at the final action
    final_action = trajectory[-1].action.action
    
    if final_action.action_type != "submit_decision":
        return 0.01 # Failed to submit a decision
        
    if final_action.decision == "CLEAR":
        return 1.0 # Correctly identified the False Positive
        
    return 0.1 # Incorrectly flagged an innocent company