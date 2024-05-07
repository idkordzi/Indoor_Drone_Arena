echo "" > $4;
echo "auto eth0" >> $4;
echo "iface eth0 inet static" >> $4;
echo "  address $1" >> $4;
echo "  netmask $2" >> $4;
echo "  gateway $3" >> $4;
