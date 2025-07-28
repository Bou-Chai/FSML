#ifndef TRAINER_H
#define TRAINER_H

class Trainer {
public:
    virtual ~Trainer() = default;
    virtual void train() = 0;
};

#endif
