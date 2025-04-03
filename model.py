# Complete Model Integration
class SignLanguageRecognitionModel:
    def __init__(self, config=CONFIG):
        self.config = config
        self.keypoint_dim = None  # Will be set when data is loaded
        self.visual_extractor = None
        self.motion_extractor = None
        self.temporal_model = None
        self.language_adapter = None
        self.classifier = None
        self.domain_confusion = None
        self.few_shot_learner = None
        self.model = None
        
    def build(self, keypoint_dim):
        """Build the complete model"""
        self.keypoint_dim = keypoint_dim
        
        # Build components
        self.visual_extractor = build_visual_feature_extractor()
        self.motion_extractor = build_motion_feature_extractor(keypoint_dim)
        self.temporal_model = build_temporal_model()
        self.language_adapter = build_language_adapter(len(self.config['languages']))
        self.classifier = build_classifier()
        self.domain_confusion = build_domain_confusion()
        
        # Define inputs
        video_input = layers.Input(shape=(self.config['num_frames'],) + self.config['input_shape'])
        keypoint_input = layers.Input(shape=(self.config['num_frames'], keypoint_dim))
        language_input = layers.Input(shape=(len(self.config['languages'])))
        
        # Feature extraction
        visual_features = self.visual_extractor(video_input)
        motion_features = self.motion_extractor(keypoint_input)
        
        # Temporal modeling
        temporal_features = self.temporal_model([visual_features, motion_features])
        
        # Language adaptation
        adapted_features = self.language_adapter([temporal_features, language_input])
        
        # Classification
        class_outputs = self.classifier(adapted_features)
        
        # Domain confusion (adversarial)
        domain_outputs = self.domain_confusion(temporal_features)
        
        # Create model
        self.model = Model(
            inputs=[video_input, keypoint_input, language_input],
            outputs=[class_outputs, domain_outputs],
            name="sign_language_recognition"
        )
        
        # Compile model with multi-task loss
        self.model.compile(
            optimizer=optimizers.Adam(self.config['learning_rate']),
            loss={
                'classifier': 'sparse_categorical_crossentropy',
                'domain_confusion': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'classifier': 1.0,
                'domain_confusion': 0.2  # Lower weight for domain confusion
            },
            metrics={
                'classifier': 'accuracy',
                'domain_confusion': 'accuracy'
            }
        )
        
        # Initialize few-shot learning component
        self.few_shot_learner = PrototypicalNetworks(self)
        
        return self.model
    
    def get_features(self, video_batch, keypoint_batch, language_batch):
        """Extract features from the model"""
        # Visual features
        visual_features = self.visual_extractor.predict(video_batch)
        
        # Motion features
        motion_features = self.motion_extractor.predict(keypoint_batch)
        
        return visual_features, motion_features
    
    def train(self, train_videos, train_keypoints, train_labels, train_languages,
             val_videos=None, val_keypoints=None, val_labels=None, val_languages=None):
        """Train the model"""
        
        # Convert language IDs to one-hot encoding
        train_languages_onehot = tf.one_hot(train_languages, depth=len(self.config['languages']))
        
        # Prepare callbacks
        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_classifier_accuracy'),
            EarlyStopping(monitor='val_classifier_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_classifier_accuracy', factor=0.5, patience=5)
        ]
        
        # Validation data
        validation_data = None
        if val_videos is not None:
            val_languages_onehot = tf.one_hot(val_languages, depth=len(self.config['languages']))
            validation_data = (
                [val_videos, val_keypoints, val_languages_onehot],
                [val_labels, val_languages]
            )
        
        # Train the model
        history = self.model.fit(
            [train_videos, train_keypoints, train_languages_onehot],
            [train_labels, train_languages],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_videos, test_keypoints, test_labels, test_languages):
        """Evaluate the model"""
        # Convert language IDs to one-hot encoding
        test_languages_onehot = tf.one_hot(test_languages, depth=len(self.config['languages']))
        
        # Evaluate
        results = self.model.evaluate(
            [test_videos, test_keypoints, test_languages_onehot],
            [test_labels, test_languages],
            batch_size=self.config['batch_size']
        )
        
        # Return metrics
        metrics = {
            'loss': results[0],
            'classifier_loss': results[1],
            'domain_confusion_loss': results[2],
            'classifier_accuracy': results[3],
            'domain_confusion_accuracy': results[4]
        }
        
        return metrics
    
    def predict(self, video, keypoints, language_id):
        """Predict class for a single video"""
        # Convert to batch format
        video_batch = np.expand_dims(video, axis=0)
        keypoint_batch = np.expand_dims(keypoints, axis=0)
        language_onehot = tf.one_hot([language_id], depth=len(self.config['languages']))
        
        # Predict
        class_probs, _ = self.model.predict([video_batch, keypoint_batch, language_onehot])
        
        # Get predicted class
        predicted_class = np.argmax(class_probs[0])
        confidence = class_probs[0][predicted_class]
        
        return predicted_class, confidence
    
    def save(self, filepath):
        """Save the model"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'GradientReversalLayer': GradientReversalLayer
        })
