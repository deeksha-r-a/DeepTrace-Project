# Setup a simple dummy model
model = nn.Linear(10, 2)

# Configure AdamW Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Configure CosineAnnealingLR Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Verify Configuration
print(f"Initial LR: {optimizer.param_groups[0]['lr']}")

# Dummy optimization and scheduler step
optimizer.step()
scheduler.step()

print(f"LR after one step: {optimizer.param_groups[0]['lr']}")
