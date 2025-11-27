import numpy as np
import matplotlib.pyplot as plt

def predict_and_visualize(model, test_generator):
    class_names = list(test_generator.class_indices.keys())
    class_indices = {c: None for c in range(len(class_names))}
    images_list, labels_list, preds_list = [], [], []

    print("Fetching test batches for prediction...")
    for images, labels in test_generator:
        preds = model.predict(images, verbose=0)
        images_list.append(images)
        labels_list.append(labels)
        preds_list.append(preds)

        for i in range(len(labels)):
            c = np.argmax(labels[i])
            if class_indices[c] is None:
                class_indices[c] = (len(images_list) - 1, i)
        if all(v is not None for v in class_indices.values()):
            break

    for c, idx in class_indices.items():
        if idx is None:
            print(f"Warning: No sample for class {class_names[c]}")
            continue
        batch_idx, sample_idx = idx
        img = images_list[batch_idx][sample_idx]
        if img.max() > 1.0:
            img = img / 255.0
        true = class_names[np.argmax(labels_list[batch_idx][sample_idx])]
        pred = class_names[np.argmax(preds_list[batch_idx][sample_idx])]

        plt.figure()
        plt.imshow(img)
        plt.title(f'Pred: {pred} | True: {true}')
        plt.axis('off')
        save_name = f'prediction_result_Tr-{class_names[c][:2]}Tr_0000.jpg'
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")