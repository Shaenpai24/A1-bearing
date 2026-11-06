# Week-1 Timed Reference Run --- Paderborn A1

## Run configuration

  Item                 Result
  -------------------- ------------------------------------------------
  Seed                 22
  Threads              6
  DataLoader workers   4
  Batch size           64
  Learning rate        0.001
  Epoch limit          30
  Elapsed time limit   30 minutes
  PyTorch              2.14.1+cpu
  Python               3.14.2
  OS                   Linux 6.17.0-35-generic-x86_64 with glibc 2.39
  CPU architecture     x86_64
  CPU cores visible    16

## Dataset

  Partition      Windows   NPZ files
  ------------ --------- -----------
  Train            3,116         100
  Validation       1,863          60
  Test             3,724         120

## Model

**Small1DCNN**

-   Raw vibration input
-   Trainable parameters: **56,131**
-   3-class output

Architecture:

``` text
Conv1d(1, 32, kernel_size=7)
ReLU
MaxPool1d(2)
Conv1d(32, 64, kernel_size=7)
ReLU
MaxPool1d(2)
Conv1d(64, 128, kernel_size=5)
ReLU
AdaptiveAvgPool1d(1)
Linear(128 -> 3)
```

## Epoch results

  --------------------------------------------------------------------------------
       Epoch Train loss   Validation      Train   Validation Epoch time    Updates
                                loss   accuracy     accuracy
  ---------- ---------- ------------ ---------- ------------ ---------- ----------
           1   0.550008     3.880205     0.7689       0.3328     8.46 s         49

           2   0.007959     5.197725     1.0000       0.3328     6.97 s         98

           3   0.001958     5.698466     1.0000       0.3328     6.95 s        147

           4   0.000940     6.045399     1.0000       0.3328     7.04 s        196

           5   0.000536     6.403872     1.0000       0.3328     7.06 s        245

           6   0.000363     6.492560     1.0000       0.3328     7.29 s        294
  --------------------------------------------------------------------------------

## Timing and resource measurements

  Measurement                                        Result
  ----------------------------- ---------------------------
  Stop reason                                Early stopping
  Epochs completed                                        6
  Optimization updates                                  294
  Fit elapsed                                  **43.772 s**
  External wall-clock elapsed                   **55.98 s**
  User CPU time                                **263.48 s**
  System CPU time                               **12.11 s**
  Aggregate CPU time                           **275.59 s**
  CPU utilisation                                      492%
  Peak resident memory            **661,544 kB (\~646 MB)**
  Major page faults                                      17
  Minor page faults                               6,600,821
  Swaps                                                   0
  Exit status                                             0

## Training behaviour

-   Training loss moved: **Yes**
-   Validation loss moved: **Yes**
-   Best validation loss: **3.880205**
-   Training accuracy reached **100% at epoch 2**
-   Validation accuracy remained **33.28%**
-   Validation loss increased after epoch 1.
-   The run stopped after five validation checks without improvement.
-   The 30-minute elapsed-time ceiling was **not reached**.

## Inference

-   Inference windows: **3,724**
-   Inference elapsed: **2.013 s**
-   Test labels and test metrics were **not evaluated**.

## Output files

``` text
runs/paderborn_week1/best_validation_checkpoint.pt
runs/paderborn_week1/timing_report.json
```

## Readiness-note summary

The required Week-1 Paderborn reference fit completed successfully under
the prescribed configuration. The run completed 6 epochs and 294
optimization updates before validation early stopping. Reported fit time
was 43.772 s, with 55.98 s external wall-clock time, 275.59 s aggregate
CPU time, and approximately 646 MB peak resident memory. Training loss
decreased substantially, while validation loss worsened after the first
epoch and validation accuracy remained at 33.28%. The run therefore
documents rapid training-set fitting with no corresponding validation
improvement under the fixed bearing-disjoint split.

## Provenance

These values are transcribed from the completed terminal output of:

``` text
/usr/bin/time -v python scripts/train_paderborn_1d_week1.py
```

No test metrics were calculated in this Week-1 timing run.
