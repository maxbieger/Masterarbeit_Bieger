Attantion nach dense damit es sich auf die neue netzwerk bezieht und nicht das alte netzwerk

#Notiz Grid Search funktioniert für Resnet nicht

# Abbruchbedingungen: Die Zahlen-Kombination muss groß genug sein: 
# Epochen=3 => (Layer1=32 Layer2=16) geht
# Epochen=3 => (Layer1=32 Layer2=4) geht nicht
# Epochen=10 => (Layer1=8 Layer2=4) geht nicht
# Epochen=20 => (Layer1=8 Layer2=4) geht nicht
# Epochen=20 => (Layer1=8 Layer2=8) geht 
# Epochen=10 => (Layer1=32 Layer2=32) geht
# Epochen=100 => (Layer1=6 Layer2=4) geht nicht
# Epochen=200 => (Layer1=6 Layer2=4) geht nicht
# Epochen=200 => (Layer1=6 Layer2=6) geht nicht
# Epochen=100 => (Layer1=8 Layer2=6) geht nicht
# Epochen=100 => (Layer1=10 Layer2=6) geht nicht
# Epochen=100 => (Layer1=10 Layer2=8) geht
# Epochen=5 => (Layer1=16 Layer2=8) geht
# Epochen=5 => (Layer1=12 Layer2=4) geht nicht
# Epochen=15 => (Layer1=12 Layer2=4) geht nicht
# Epochen=5 => (Layer1=16 Layer2=8) 170, geht 
# Epochen=5 => (Layer1=16 Layer2=8) 170, 50V2, geht über 50% acc
# Epochen=5 => (Layer1=16 Layer2=8) 200, 50V2, cbam_block, l2(0.05), Dropout hoch, geht nicht
# Epochen=5 => (Layer1=16 Layer2=8) 200, 50V2, l2(0.05), Dropout hoch, geht 61% acc
# Epochen=10 => (Layer1=10 Layer2=8) 185, 50V2, l2(0.05), Dropout hoch, ResNet_a, 71%
# Epochen=10 => (Layer1=10 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 84%
# Epochen=5 => (Layer1=10 Layer2=8) 170, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 64%
# Epochen=10 => (Layer1=12 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 78%
# # Epochen=10 => (Layer1=12 Layer2=8) 185, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 77%
# Epochen=50 => (Layer1=10 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 94%
# Epochen=50 => (Layer1=10 Layer2=8) 180, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a", 71%
#Epochen=50 => (Layer1=10 Layer2=8) 182, 50V2, l2(0.04), Dropout 0.5/0.4, ResNet_a, 73%
# Epochen=50 => (Layer1=8 Layer2=8) 185, 50V2, l2(0.05), Dropout 0.5/0.4, ResNet_a, 90%
# Epochen=60 => (Layer1=8 Layer2=8) 187, 50V2, l2(0.1), Dropout 0.5/0.5, ResNet_a, 89%
# Epochen=60 => (Layer1=8 Layer2=8) 190, 50V2, l2(0.1), Dropout 0.5/0.5, ResNet_a, 79% weniger overfitting"
# "Epochen=60 => (Layer1=8 Layer2=8) 185, 50V2, l2(0.1), Dropout 0.5/0.5, ResNet_a, noch machen"