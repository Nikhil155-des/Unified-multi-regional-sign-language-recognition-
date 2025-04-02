# Define the Visual Feature Extraction Module
def build_visual_feature_extractor():
    """Build the visual feature extraction using MobileNetV3"""
    base_model = MobileNetV3Large(
        input_shape=CONFIG['input_shape'],
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Build model
    inputs = layers.Input(shape=(CONFIG['num_frames'],) + CONFIG['input_shape'])
    
    # Apply the base model to each frame
    x = layers.TimeDistributed(base_model)(inputs)
    
    # Add a dense layer to reduce dimensions
    x = layers.TimeDistributed(layers.Dense(CONFIG['visual_feature_dim'], activation='relu'))(x)
    
    model = Model(inputs=inputs, outputs=x, name="visual_extractor")
    return model

# Define the Motion Feature Extraction Module
def build_motion_feature_extractor(keypoint_dim):
    """Build the motion/pose feature extraction using TCN"""
    inputs = layers.Input(shape=(CONFIG['num_frames'], keypoint_dim))
    
    # 1D Temporal Convolutional Network
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(CONFIG['motion_feature_dim'], kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    model = Model(inputs=inputs, outputs=x, name="motion_extractor")
    return model

# Define a Gated Recurrent Unit (GRU) for temporal modeling
def build_temporal_model():
    """Build the temporal model using GRU"""
    # Visual feature input
    visual_input = layers.Input(shape=(CONFIG['num_frames'], CONFIG['visual_feature_dim']))
    
    # Motion feature input
    motion_input = layers.Input(shape=(CONFIG['num_frames'], CONFIG['motion_feature_dim']))
    
    # Concatenate features
    combined = layers.Concatenate()([visual_input, motion_input])
    
    # Bidirectional GRU for temporal modeling
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(combined)
    x = layers.Bidirectional(layers.GRU(256, return_sequences=False))(x)
    
    # Dense layer to generate temporal features
    temporal_features = layers.Dense(CONFIG['shared_feature_dim'], activation='relu')(x)
    
    model = Model(inputs=[visual_input, motion_input], outputs=temporal_features, name="temporal_model")
    return model
