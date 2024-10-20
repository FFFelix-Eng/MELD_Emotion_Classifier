import os
import pandas as pd


def save_predictions(model_name, save_path, id2label, predictions, labels, mask, dialogue_id, dialogues):
    df = pd.DataFrame(columns=['dialogue_id', 'utterance_id', 'utterance', 'label', 'prediction'])

    # Each Batch
    for pred_batch, label_batch, mask_batch, id_batch, dia_batch in zip(predictions, labels, mask, dialogue_id,
                                                                        dialogues):

        # print(f"DEBUGGG0: {pred_batch}")
        # print(f"DEBUGGG1: {label_batch}")
        # print(f"DEBUGGG2: {dia_batch}")
        # wrapping for single dialogues
        if isinstance(dia_batch[0][0], str):
            # If it's a list of utterances, wrap it inside another list to treat it as a single dialogue batch
            # Loader seems to change every string to a string wraped by ()
            # print("Wrap the single dialogue batch (from saving predictions)")
            # pred_batch = [pred_batch]
            label_batch = [label_batch]
            mask_batch = [mask_batch]
            id_batch = [id_batch]
            dia_batch = [dia_batch]

        # Each Dialogue in Batch
        for pred_dia, label_dia, m_dia, d_id, dia in zip(pred_batch, label_batch, mask_batch, id_batch, dia_batch):
            utt_id = 0
            # Each utterance in dialogue
            for pred_utt, label_utt, m_utt, utt in zip(pred_dia, label_dia, m_dia, dia):
                # if either mask code is 0 or label is -1, skip the sample
                # print(f"pred: {pred_utt}\nlabel: {label_utt}\nmask: {m_utt}\nutt:{utt}")
                label_utt = label_utt.item()
                m_utt = m_utt.item()

                if m_utt == 1 and label_utt != -1:
                    utt_id += 1
                    # create a new line in dataframe
                    temp_df = pd.DataFrame({
                        'dialogue_id': d_id,
                        'utterance_id': utt_id,
                        'utterance': utt,
                        'label': id2label.get(label_utt),
                        'prediction': id2label.get(pred_utt)
                    })
                    # print(temp_df)
                    df = pd.concat([df, temp_df], ignore_index=True)
    df.to_csv(os.path.join(save_path, f"{model_name}_predictions.csv"), index=False)