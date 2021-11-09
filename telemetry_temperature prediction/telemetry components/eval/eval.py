import argparse
def eval(rf_metrics, cat_metrics):
    import joblib
    import pandas as pd
    import pprint
    
    rf_metrics=joblib.load(rf_metrics)
    cat_metrics=joblib.load(cat_metrics)

    lis = [rf_metrics,cat_metrics]
    max = 0
    for i in lis:
        accuracy = i['test']
        if accuracy > max:
            max = accuracy
            model = i['model']
            metrics = i
  
    print('Best metrics \n\n')
    pprint.pprint(metrics)

    joblib.dump(model, 'best_model')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_metrics')
    parser.add_argument('--cat_metrics')
    args = parser.parse_args()
    eval(args.rf_metrics,args.cat_metrics)