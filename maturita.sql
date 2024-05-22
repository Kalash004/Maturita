use maturita;


-- DDL
create table maturitanti(
	id int primary key auto_increment,
    jmeno varchar(50) not null,
    prijmeni varchar(50) not null,
    employee_id int
    -- FOREIGN KEY (employee_id) REFERENCES Employee(EmployeeID) ON DELETE CASCADE
);

create table trida(
	trida_id int primary key auto_increment,
    nazev varchar(10)
);

create table zak(
	zak_id int primary key auto_increment,
    jmeno varchar(15) not null,
    f_trida_id int not null,
    foreign key (f_trida_id) references trida(trida_id) on delete cascade
);


drop table maturitanti;

alter table maturitanti;
-- add Email varchar(10);
-- drop column Email;
-- modify column Email int;
-- rename column Email to NewMail;

truncate imported_table;

-- DDL end


-- DML 
-- select
-- insert
-- delete
-- update

insert into maturitanti(jmeno, prijmeni) 
values("funny", "maturita");

update maturitanti 
set jmeno = "Very fun"
where jmeno = "funny";

set sql_safe_updates=0;

delete from maturitanti
where jmeno = "Very fun";


-- DML end


-- Jazyk SQL - SELECT, VIEW (spojování tabulek, agregační funkce, seskupování záznamů)

Select id as `Identification`, concat(jmeno," ",prijmeni) as `Jmeno Prijmeni` from maturitanti;

create view maturita_view
as
Select id as `Identification`, concat(jmeno," ",prijmeni) as `Jmeno Prijmeni` from maturitanti;



select count(*) as `Count of instances` from maturita_view;

insert into trida (nazev) values ("C4B");
insert into trida (nazev) values ("C4C");
insert into zak (jmeno, f_trida_id) values ("Anton", 1);
insert into zak (jmeno, f_trida_id) values ("Cecak", 2);


select zak.zak_id, zak.jmeno, trida.nazev 
from zak 
inner join 
trida on trida.trida_id = zak.f_trida_id;


select trida.nazev ,count(*) 
from trida 
inner join zak 
on zak.f_trida_id = trida.trida_id
group by trida.nazev;

-- End

-- Jazyk SQL - Vnořené příkazy (operátory IN, EXISTS, ALL, SOME, ANY)

select trida.nazev
from trida
where trida.trida_id in (select zak.f_trida_id from zak where zak.jmeno = "Anton");

-- end

-- Jazyk SQL - DCL, TCL příkazy

-- Grant
-- Revoke

grant select on table maturitanti to maturita_user;


-- Start transaction
-- Commit 
-- Rollback
-- Savepoint 

SET autocommit = 1;

start transaction;
commit;
Rollback;
savepoint xy;
rollback to xy;


-- Indexy a indexace dat v databázi (UNIQUE, INDEX)

create index maturita_indx
on zak (jmeno);


-- Uložené procedury a funkce

delimiter //
Create procedure remove_student (in jmeno varchar(15), out msg varchar(3))
begin  
	delete from zak 
    where zak.jmeno = jmeno;
End //

call remove_student("cecak", @msg);
select @msg;


create function hello (jmeno varchar(15)) 
returns varchar(50) deterministic
return concat("Hello, ",jmeno," !");

select hello(jmeno) from zak;


-- end 

-- Triggery

create table zak_log(
	zak_log_id int primary key auto_increment,
    jmeno varchar(15) not null,
    f_trida_id int not null
);


delimiter //
create trigger delete_log
after delete on zak
for each row
begin
insert into zak_log (jmeno, f_trida_id) values(old.jmeno, old.f_trida_id);
end; // 

delete from zak
where jmeno = "Anton";
