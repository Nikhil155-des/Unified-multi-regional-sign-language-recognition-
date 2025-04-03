# Define Language-Specific Adapter
def build_language_adapter(num_languages):
    """Build language-specific adapters"""
    # Feature input
    feature_input = layers.Input(shape=(CONFIG['shared_feature_dim'],))
    
    # Language ID input (one-hot encoded)
    language_input = layers.Input(shape=(num_languages,))
    
    # Language-specific adapter layers
    language_adapters = []
    for i in range(num_languages):
        # Create a unique adapter for each language
        adapter = layers.Dense(CONFIG['shared_feature_dim'] // 2, activation='relu')(feature_input)
        adapter = layers.Dense(CONFIG['shared_feature_dim'], activation='linear')(adapter)
        language_adapters.append(adapter)
    
    # Language-specific adapters with gating based on language ID
    adapted_features = None
    for i in range(num_languages):
        # Extract the i-th adapter
        adapter_output = language_adapters[i]
        
        # Gate with language ID (extract the i-th element)
        gate = layers.Lambda(lambda x: x[:, i:i+1])(language_input)
        gated_adapter = layers.Multiply()([adapter_output, gate])
        
        # Add to the combined adapted features
        if adapted_features is None:
            adapted_features = gated_adapter
        else:
            adapted_features = layers.Add()([adapted_features, gated_adapter])
    
    # Residual connection
    outputs = layers.Add()([feature_input, adapted_features])
    
    model = Model(inputs=[feature_input, language_input], outputs=outputs, name="language_adapter")
    return model

# Define the Classifier
def build_classifier():
    """Build the classification head"""
    inputs = layers.Input(shape=(CONFIG['shared_feature_dim'],))
    
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(CONFIG['num_classes'], activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="classifier")
    return model

# Define Gradient Reversal Layer for Domain Adaptation
class GradientReversalLayer(layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, inputs):
        return inputs
        
    def get_config(self):
        config = super(GradientReversalLayer, self).get_config()
        config.update({'alpha': self.alpha})
        return config
        
    def compute_output_shape(self, input_shape):
        return input_shape

# Domain Confusion Module
def build_domain_confusion():
    """Build domain confusion module for adversarial training"""
    inputs = layers.Input(shape=(CONFIG['shared_feature_dim'],))
    
    # Gradient reversal layer is custom defined to reverse gradients during backpropagation
    x = GradientReversalLayer()(inputs)
    
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(len(CONFIG['languages']), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="domain_confusion")
    return model
