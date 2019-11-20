def fit(fcm, windows, batch_size=1, lr=0.001, savefolder="run0",
        X_test=None, y_test=None,
        X_train=None, y_train=None,
        weight_decay=0.01):
    train_dataset = PermutedSubsampledCorpus(windows)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True,
                                                   drop_last=False)

    if fcm.gpu:
        fcm.cuda()

    train_loss_file = open(savefolder + "/train_loss.txt", "w")

    # SGD generalizes better: https://arxiv.org/abs/1705.08292
    optimizer = optim.Adam(fcm.parameters(), lr=lr, weight_decay=weight_decay)
    nwindows = len(windows)

    for epoch in tqdm(range(fcm.nepochs), desc="Training epochs"):
        total_sgns_loss = 0.0
        total_dirichlet_loss = 0.0
        total_pred_loss = 0.0
        total_diversity_loss = 0.0

        fcm.train()
        for batch in train_dataloader:
            loss, sgns_loss, dirichlet_loss, pred_loss, div_loss = fcm.calculate_loss(batch)

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            for p in fcm.parameters():
                if p.requires_grad:
                    p.grad = p.grad.clamp(min=-GRAD_CLIP, max=GRAD_CLIP)

            optimizer.step()

            nsamples = batch.size(0)

            total_sgns_loss += sgns_loss.data * nsamples
            total_dirichlet_loss += dirichlet_loss.data * nsamples
            total_pred_loss += pred_loss.data * nsamples
            total_diversity_loss += div_loss.data * nsamples

        auc = -1.
        if X_test is not None and y_test is not None:
            if fcm.expvars_test is None:
                y_pred = fcm.predict_proba(torch.FloatTensor(X_test)).cpu().detach().numpy()
            else:
                y_pred = fcm.predict_proba(torch.FloatTensor(X_test),
                                           fcm.expvars_test).cpu().detach().numpy()
            auc = roc_auc_score(y_test, y_pred)
        train_auc = -1.
        if X_train is not None and y_train is not None:
            if fcm.expvars_train is None:
                y_pred = fcm.predict_proba(torch.FloatTensor(X_train)).cpu().detach().numpy()
            else:
                y_pred = fcm.predict_proba(torch.FloatTensor(X_train),
                                           fcm.expvars_train).cpu().detach().numpy()
            train_auc = roc_auc_score(y_train, y_pred)

        print("epoch " + str(epoch) + ":")
        print("Train AUC: %.4f" % (train_auc))
        print("Test AUC: %.4f" % (auc)),
        print("Total loss: %.4f" % (
                    (total_sgns_loss + total_dirichlet_loss + total_pred_loss + total_diversity_loss) / nwindows))
        print("SGNS loss: %.4f" % (total_sgns_loss / nwindows))
        print("Dirichlet loss: %.4f" % (total_dirichlet_loss / nwindows))
        print("Prediction loss: %.4f" % (total_pred_loss / nwindows))
        print("Diversity loss: %.4f" % (total_diversity_loss / nwindows))
        train_loss_file.write("%.4f" % (
                    (total_sgns_loss + total_dirichlet_loss + total_pred_loss + total_diversity_loss) / nwindows) + ",")
        train_loss_file.write("%.4f" % (total_sgns_loss / nwindows) + ",")
        train_loss_file.write("%.4f" % (total_dirichlet_loss / nwindows) + ",")
        train_loss_file.write("%.4f" % (total_pred_loss / nwindows) + ",")
        train_loss_file.write("%.4f" % (total_diversity_loss / nwindows) + "\n")
        train_loss_file.flush()

        if (epoch + 1) % 10 == 0:
            torch.save(fcm.state_dict(), savefolder + "/" + str(epoch + 1) + ".slda2vec.pytorch")

    # TODO: save the best on valid?
    torch.save(fcm.state_dict(), savefolder + "/slda2vec.pytorch")