from src import preprocess_data as pre

t = True
f = False

PREPROCESS = f


def main():
    if PREPROCESS:
        dataset = pre.transform_jpg_to_dataset(
            "/Users/aidenyoung/pythonshi/dataproj/data/raw/Expressions")
        pre.save_dataset(dataset, 
                    "/Users/aidenyoung/pythonshi/dataproj/data/processed/expression_tensors.pt")
        data = pre.load_dataset(
        "/Users/aidenyoung/pythonshi/dataproj/data/processed/expression_tensors.pt")
        pre.create_train_test_split(data,
                            "/Users/aidenyoung/pythonshi/dataproj/data/processed/train_test.pt")

#/////////////////////////////////////////////////////////////////////////////////////////////////////

    data = pre.load_dataset(
                       "/Users/aidenyoung/pythonshi/dataproj/data/processed/train_test.pt")
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    class_to_idx = data["class_to_idx"]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(class_to_idx)

if __name__ == '__main__':
    main()