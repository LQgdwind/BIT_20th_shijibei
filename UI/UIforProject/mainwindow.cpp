#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "stdio.h"
#include "QProcess"
#include "QMessageBox"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("城市垃圾分类管理系统");
    setFixedSize(400,300);
    QPalette pal = this->palette();
    pal.setBrush(QPalette::Background,QBrush(QPixmap("../picture/1.png")));
    setPalette(pal);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QProcess p(0);
    QString command = "../../source_code/train.py";
    p.execute(command);
    p.waitForFinished();
    QMessageBox MyBox(QMessageBox::Question,"提示","请在本地正确安装cuda和pytorch环境",QMessageBox::Yes);
    MyBox.exec();
}

void MainWindow::on_pushButton_2_clicked()
{
    QProcess p(0);
    QString command = "../../source_code/prediction.py";
    p.execute(command);//command是要执行的命令,args是参数
    p.waitForFinished();
    QMessageBox MyBox(QMessageBox::Question,"提示","请在本地正确安装cuda和pytorch环境",QMessageBox::Yes);
    MyBox.exec();
}
