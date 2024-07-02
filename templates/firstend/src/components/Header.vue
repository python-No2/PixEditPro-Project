<template>
    <div class="header-content">
        <span class="Logo">PixEditPro</span>
        <nav class="navbar">
            <router-link to="/home" class="nav-item" active-class="active-link">Home</router-link>
            <router-link to="/basic" class="nav-item" active-class="active-link">Basic</router-link>
            <router-link to="/restoration" class="nav-item" active-class="active-link">Restoration</router-link>
            <router-link to="/transfer" class="nav-item" active-class="active-link">Transfer</router-link>
        </nav>
    </div>
</template>
<script setup>
</script>
<style>
.header-content {
    display: flex;
    flex-direction: row;
    position: fixed;
    left: 50px;
    top: 30px;
}
.Logo{
    color: #FFFFFF;
    font-size: 30px;
    font-weight: bold;
    font-style: inherit;
}
.navbar {
    position: fixed;
    left: 250px;
    top: 20px;
    display: flex;
    background-color: transparent;
    padding: 10px;
}

.nav-item {
    color: #ffffff;
    text-decoration: none;
    padding: 10px;
    transition: color,font-size 0.3s;
    font-size: 20px;
    font-weight: bold;
}

.nav-item:hover {
    color: #ffd04b;
    font-size: 25px;
}

.nav-item.active {
    color: #ffd04b;

}
.active-link {
    color: #ffd04b;
    font-size: 25px;
}
</style>
